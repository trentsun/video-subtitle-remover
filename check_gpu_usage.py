#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查模型GPU使用情况的工具
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
import subprocess
import platform
from pprint import pprint
import importlib.util
from pathlib import Path

# 导入后端模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend.config as config
from backend.main import SubtitleDetect, SubtitleRemover

def print_title(title):
    """打印带格式的标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_gpu_availability():
    """检查系统GPU是否可用"""
    print_title("系统GPU检查")
    
    # 检查NVIDIA驱动
    if platform.system() == 'Windows':
        try:
            subprocess.check_output('nvidia-smi')
            print("✅ NVIDIA驱动已安装并正常运行")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ NVIDIA驱动未安装或运行异常")
    else:  # Linux/Mac
        try:
            subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
            print("✅ NVIDIA驱动已安装并正常运行")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ NVIDIA驱动未安装或运行异常")
    
    # 检查PyTorch GPU支持
    print("\n[PyTorch GPU支持]")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存总量: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"    计算能力: {torch.cuda.get_device_capability(i)}")
    else:
        print("❌ PyTorch未检测到CUDA设备，请检查CUDA安装")

def check_onnx_runtime():
    """检查ONNX Runtime GPU支持"""
    print_title("ONNX Runtime GPU支持检查")
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime版本: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"可用提供商: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime已启用CUDA支持")
            sess_options = ort.SessionOptions()
            print("\n尝试创建带CUDA提供商的会话...")
            try:
                # 创建一个临时会话测试CUDA提供商
                temp_sess = ort.InferenceSession(
                    "dummy_path",
                    sess_options=sess_options,
                    providers=['CUDAExecutionProvider'],
                )
                print("✅ 成功初始化CUDA提供商")
            except Exception as e:
                if "No such file or directory" in str(e):
                    print("✅ CUDA提供商配置正确（忽略'No such file'错误）")
                else:
                    print(f"❌ CUDA提供商初始化失败: {str(e)}")
        else:
            print("❌ ONNX Runtime未启用CUDA支持")
    except ImportError:
        print("❌ 未安装ONNX Runtime，请使用 pip install onnxruntime-gpu 安装")

def check_paddle_gpu():
    """检查PaddlePaddle GPU支持"""
    print_title("PaddlePaddle GPU支持检查")
    
    try:
        import paddle
        print(f"PaddlePaddle版本: {paddle.__version__}")
        
        if paddle.is_compiled_with_cuda():
            print("✅ PaddlePaddle已编译支持CUDA")
            print(f"可用GPU设备: {paddle.device.get_available_device()}")
            
            # 检查是否实际可用
            try:
                x = paddle.to_tensor([1.0, 2.0, 3.0], place=paddle.CUDAPlace(0))
                print("✅ 成功在GPU上创建Paddle张量")
            except Exception as e:
                print(f"❌ GPU张量创建失败: {str(e)}")
        else:
            print("❌ PaddlePaddle未编译支持CUDA")
    except ImportError:
        print("❓ 未安装PaddlePaddle，或未被项目使用")

def check_subtitle_detector(video_path):
    """检查字幕检测器是否正确使用GPU"""
    print_title("字幕检测器GPU使用检查")
    start_time = time.time()
    
    try:
        # 创建字幕检测对象
        detector = SubtitleDetect(video_path)
        
        # 打印当前设置
        print(f"USE_ONNX: {hasattr(config, 'USE_ONNX') and config.USE_ONNX}")
        print(f"ONNX提供商: {config.ONNX_PROVIDERS}")
        
        # 执行一次字幕检测测试GPU使用
        import cv2
        video_cap = cv2.VideoCapture(video_path)
        ret, frame = video_cap.read()
        if ret:
            print("正在进行字幕检测测试...")
            dt_boxes, elapse = detector.detect_subtitle(frame)
            print(f"字幕检测耗时: {elapse:.4f}秒")
            print(f"检测到 {len(dt_boxes)} 个文本框")
        else:
            print("❌ 无法读取视频帧进行测试")
        
        video_cap.release()
        
        # 检查ONNX模型转换
        model_dir = config.DET_MODEL_PATH
        onnx_model_path = os.path.join(model_dir, "model.onnx")
        if os.path.exists(onnx_model_path):
            print(f"✅ 检测到转换的ONNX模型: {onnx_model_path}")
        else:
            print(f"❓ 未找到ONNX模型，可能使用原始Paddle模型")
        
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        
        # 返回成功结果
        return True
    except Exception as e:
        print(f"❌ 字幕检测器测试出错: {e}")
        return False

def check_full_pipeline(video_path):
    """检查完整处理流程的GPU使用情况"""
    print_title("完整处理流程GPU使用检查")
    print(f"视频路径: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    start_time = time.time()
    
    try:
        # 监控GPU使用情况
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"初始GPU内存使用: {initial_mem:.2f} MB")
        
        # 创建SubtitleRemover对象
        remover = SubtitleRemover(video_path, gui_mode=False)
        
        # 测试单帧处理
        if hasattr(remover, "video_cap") and remover.video_cap.isOpened():
            ret, frame = remover.video_cap.read()
            if ret:
                # 检查字幕帧寻找
                print("正在测试字幕帧寻找，这可能需要一些时间...")
                subtitle_frames = remover.sub_detector.find_subtitle_frame_no()
                
                if subtitle_frames:
                    print(f"✅ 成功找到 {len(subtitle_frames)} 个包含字幕的帧")
                else:
                    print("⚠️ 未找到任何包含字幕的帧，视频可能没有字幕或检测失败")
            else:
                print("❌ 无法读取视频帧进行测试")
        else:
            print("❌ 无法打开视频进行测试")
        
        # 报告内存使用
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"峰值GPU内存使用: {peak_mem:.2f} MB")
            print(f"当前GPU内存使用: {current_mem:.2f} MB")
            
            if peak_mem > initial_mem + 10:  # 至少使用了10MB以上显存
                print("✅ 检测到明显的GPU内存使用，处理流程正在使用GPU")
            else:
                print("⚠️ 未检测到明显的GPU内存使用，处理流程可能未充分利用GPU")
        
        print(f"总耗时: {time.time() - start_time:.2f}秒")
        
        return True
    except Exception as e:
        print(f"❌ 完整处理流程测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detector_gpu(video_path=None):
    """测试文本检测器是否使用GPU"""
    print_title("文本检测器GPU使用检查")
    
    try:
        # 动态导入项目模块
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        from sub_detector import SubtitleDetect
        
        print(f"ONNX启用状态: {getattr(config, 'USE_ONNX', False)}")
        
        # 创建检测器
        print("正在初始化文本检测器...")
        detector = SubtitleDetect()
        
        # 如果提供了视频，测试实际处理
        if video_path and os.path.exists(video_path):
            import cv2
            print(f"使用视频文件测试: {video_path}")
            
            # 读取视频帧
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # 记录内存使用前GPU使用
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()
                
                # 运行检测
                start_time = time.time()
                result = detector.text_detector(frame)
                end_time = time.time()
                
                print(f"检测耗时: {end_time - start_time:.4f}秒")
                
                # 检查GPU内存变化
                if torch.cuda.is_available():
                    end_mem = torch.cuda.memory_allocated()
                    delta = end_mem - start_mem
                    
                    if delta > 0:
                        print(f"✅ 检测过程使用了GPU内存: {delta / (1024**2):.2f} MB")
                    else:
                        print("❌ 检测过程没有使用GPU内存，可能运行在CPU上")
                
                print(f"检测到的文本区域数量: {len(result) if result else 0}")
            else:
                print(f"❌ 无法读取视频帧: {video_path}")
        else:
            print("未提供有效视频文件，跳过实际检测测试")
            
    except ImportError as e:
        print(f"❌ 导入检测器模块失败: {str(e)}")
    except Exception as e:
        print(f"❌ 测试文本检测器失败: {str(e)}")

def check_config_file():
    """检查配置文件中的关键设置"""
    print_title("配置文件检查")
    
    try:
        # 动态导入项目模块
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        print("关键配置项:")
        print(f"  USE_ONNX = {getattr(config, 'USE_ONNX', False)}")
        print(f"  DEVICE = {getattr(config, 'DEVICE', 'cpu')}")
        print(f"  EP_LIST = {getattr(config, 'EP_LIST', [])}")
        print(f"  PROPAINTER_MAX_LOAD_NUM = {getattr(config, 'PROPAINTER_MAX_LOAD_NUM', 'Not Set')}")
        
        # 提供优化建议
        if not getattr(config, 'USE_ONNX', False):
            print("\n❗建议: 在config.py中设置 USE_ONNX = True 以启用ONNX加速")
        
        if 'CUDAExecutionProvider' not in getattr(config, 'EP_LIST', []):
            print("❗建议: 确保EP_LIST中包含'CUDAExecutionProvider'")
        
    except Exception as e:
        print(f"❌ 检查配置文件失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='检查视频字幕移除工具的GPU加速状态')
    parser.add_argument('--video', type=str, help='用于测试的视频文件路径')
    args = parser.parse_args()
    
    print("\n🔍 开始GPU加速检查...\n")
    
    check_gpu_availability()
    check_onnx_runtime()  
    check_paddle_gpu()
    check_config_file()
    
    if args.video:
        test_detector_gpu(args.video)
    else:
        print("\n⚠️ 未提供视频文件，跳过文本检测器测试")
        print("要进行完整测试，请指定视频文件：python check_gpu_usage.py --video your_video.mp4")
    
    print("\n✅ 检查完成！请查看以上信息确认GPU加速状态")

if __name__ == "__main__":
    main() 