# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

import torch
import torchvision

from backend import config
from backend.inpaint.video.model.modules.flow_comp_raft import RAFT_bi
from backend.inpaint.video.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from backend.inpaint.video.model.propainter import InpaintGenerator
from backend.inpaint.video.core.utils import to_tensors
from backend.inpaint.video.model.misc import get_device

import warnings

warnings.filterwarnings("ignore")


def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    # 如果传入的直接为numpy array
    if isinstance(mpath, np.ndarray):
        masks_img = [Image.fromarray(mpath)]
    # input single img path
    else:
        if isinstance(mpath, str):
            if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
                masks_img = [Image.open(mpath)]
        else:
            mnames = sorted(os.listdir(mpath))
            for mp in mnames:
                masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros((imgH_extr, imgW_extr, 3), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

    mask[H_start + dilate_h: H_start + imgH - dilate_h,
    W_start + dilate_w: W_start + imgW - dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class VideoInpaint:
    def __init__(self, sub_video_length=config.PROPAINTER_MAX_LOAD_NUM, use_fp16=True, scale_factor=0.5):
        self.device = get_device()
        self.use_fp16 = use_fp16
        self.use_half = True if self.use_fp16 else False
        self.scale_factor = scale_factor
        if self.device == torch.device('cpu'):
            self.use_half = False
        # Length of sub-video for long video inference.
        self.sub_video_length = sub_video_length
        # Length of local neighboring frames
        self.neighbor_length = 10
        # Mask dilation for video and flow masking
        self.mask_dilation = 4
        # Stride of global reference frames
        self.ref_stride = 10
        # Iterations for RAFT inference
        self.raft_iter = 20
        # Stride of global reference frames
        self.ref_stride = 10
        # 设置raft模型
        self.fix_raft = self.init_raft_model()
        # 设置fix_flow模型
        self.fix_flow_complete = self.init_fix_flow_model()
        # 设置inpaint模型
        self.model = self.init_inpaint_model()

    def downsample_frame(self, frame):
        """下采样一帧图像"""
        if self.scale_factor == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        
    def upsample_frame(self, frame, original_size):
        """上采样一帧图像"""
        if self.scale_factor == 1.0:
            return frame
        return cv2.resize(frame, original_size, interpolation=cv2.INTER_LANCZOS4)

    def init_raft_model(self):
        # set up RAFT and flow competition model
        return RAFT_bi(os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'raft-things.pth'), self.device)

    def init_fix_flow_model(self):
        fix_flow_complete_model = RecurrentFlowCompleteNet(
            os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'recurrent_flow_completion.pth'))
        for p in fix_flow_complete_model.parameters():
            p.requires_grad = False
        fix_flow_complete_model.to(self.device)
        fix_flow_complete_model.eval()
        return fix_flow_complete_model

    def init_inpaint_model(self):
        # set up ProPainter model
        return InpaintGenerator(model_path=os.path.join(config.VIDEO_INPAINT_MODEL_PATH, 'ProPainter.pth')).to(
            self.device).eval()

    def process_batch(self, frames, mask, batch_size=2):
        """处理一批帧，确保批大小合适"""
        if len(frames) <= batch_size:
            return self.process_single_batch(frames, mask)
            
        results = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self.process_single_batch(batch_frames, mask)
            results.extend(batch_results)
        return results

    def process_single_batch(self, frames, mask):
        """处理单个批次的帧"""
        if isinstance(frames[0], np.ndarray):
            # 下采样处理
            if self.scale_factor != 1.0:
                frames = [self.downsample_frame(f) for f in frames]
                mask = cv2.resize(mask, (frames[0].shape[1], frames[0].shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            
        size = frames[0].size
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask, frames_len, size,
                                            flow_mask_dilates=self.mask_dilation,
                                            mask_dilates=self.mask_dilation)
        
        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        
        frames = frames.to(self.device)
        flow_masks = flow_masks.to(self.device)
        masks_dilated = masks_dilated.to(self.device)
        
        with torch.no_grad():
            try:
                # 计算光流
                gt_flows_bi = self.fix_raft(frames, iters=self.raft_iter)
                
                # 完成光流
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                
                # 图像传播
                masked_frames = frames * (1 - masks_dilated)
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                b, t, _, h, w = masks_dilated.size()
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                
                # 处理结果
                comp_frames = []
                for i in range(frames_len):
                    frame = updated_frames[0, i].cpu().permute(1, 2, 0).numpy()
                    frame = (frame + 1) / 2 * 255
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # 上采样回原始尺寸
                    if self.scale_factor != 1.0:
                        frame = self.upsample_frame(frame, (size[0], size[1]))
                        
                    comp_frames.append(frame)
                
                return comp_frames
                
            except RuntimeError as e:
                print(f"Error processing batch: {str(e)}")
                # 如果批处理失败，尝试减小批大小重试
                if len(frames) > 1:
                    print("Retrying with smaller batch size...")
                    half = len(frames) // 2
                    first_half = self.process_single_batch(frames[:half], mask)
                    second_half = self.process_single_batch(frames[half:], mask)
                    return first_half + second_half
                else:
                    raise e

    def inpaint(self, frames, mask):
        """
        修改后的inpaint方法，使用批处理来处理帧
        """
        # 确定合适的批大小
        if len(frames) > self.sub_video_length:
            batch_size = self.sub_video_length
        else:
            batch_size = min(len(frames), 2)  # 默认使用较小的批大小
            
        print(f"Processing {len(frames)} frames with batch size {batch_size}")
        return self.process_batch(frames, mask, batch_size)


def read_frames(v_path):
    video_cap = cv2.VideoCapture(v_path)
    video_frames = []
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break
        video_frames.append(frame)
    video_frames = [Image.fromarray(f) for f in video_frames]
    return video_frames


if __name__ == '__main__':
    # VideoInpaint
    video_inpaint = VideoInpaint(sub_video_length=80)
    frames = read_frames('/home/yao/Documents/Project/video-subtitle-remover/local_test/test1.mp4')
    mask = cv2.imread('/home/yao/Documents/Project/video-subtitle-remover/local_test/test1_mask.png')
    inpainted_frames = video_inpaint.inpaint(frames, mask)
    save_root = '/home/yao/Documents/Project/video-subtitle-remover/local_test/'
    video_out_path = os.path.join(save_root, 'inpaint_out.mp4')
    print("size: ", inpainted_frames[0].shape)
    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 360))
    for comp_frame in inpainted_frames:
        video_writer.write(comp_frame)
    video_writer.release()
    print(f'\nAll results are saved in {save_root}')

