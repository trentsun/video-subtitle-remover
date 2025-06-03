import cv2
import numpy as np

class SamplingProcessor:
    def __init__(self, original_size, scale_factor=0.5):
        """
        初始化采样处理器
        
        Args:
            original_size: 原始尺寸 (width, height)
            scale_factor: 缩放因子 (0-1)
        """
        self.original_size = original_size
        self.scale_factor = scale_factor
        self.processing_size = (
            int(original_size[0] * scale_factor),
            int(original_size[1] * scale_factor)
        )
        
    def downsample_frame(self, frame):
        """下采样一帧图像"""
        return cv2.resize(frame, self.processing_size, interpolation=cv2.INTER_AREA)
        
    def upsample_frame(self, frame):
        """上采样一帧图像"""
        return cv2.resize(frame, self.original_size, interpolation=cv2.INTER_LANCZOS4)
        
    def process_frames_with_sampling(self, frames, process_func, *args, **kwargs):
        """
        使用下采样-处理-上采样的流程处理多帧图像
        
        Args:
            frames: 输入帧列表
            process_func: 处理函数
            *args, **kwargs: 传递给处理函数的参数
        """
        # 下采样
        frames_small = [self.downsample_frame(f) for f in frames]
        
        # 处理
        processed_small = process_func(frames_small, *args, **kwargs)
        
        # 上采样
        return [self.upsample_frame(f) for f in processed_small]
    
    def scale_coordinates(self, coordinates):
        """
        缩放坐标
        
        Args:
            coordinates: (xmin, xmax, ymin, ymax) 格式的坐标
        """
        xmin, xmax, ymin, ymax = coordinates
        return (
            int(xmin * self.scale_factor),
            int(xmax * self.scale_factor),
            int(ymin * self.scale_factor),
            int(ymax * self.scale_factor)
        )
    
    def scale_mask(self, mask):
        """缩放mask"""
        return cv2.resize(mask, self.processing_size, interpolation=cv2.INTER_NEAREST)