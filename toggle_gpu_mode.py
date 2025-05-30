#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU加速模式切换工具
允许在不同的GPU加速方式之间切换：
1. CUDA直接加速
2. ONNX加速
3. 多GPU并行
"""

import os
import sys
import argparse
import json
import re
import importlib.util
import shutil
from pathlib import Path

# 导入后端模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend.config as config

CONFIG_FILE = "backend/config.py"
BACKUP_DIR = "config_backups"

def print_title(title):
    """打印带格式的标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def backup_config_file():
    """备份config.py文件"""
    if os.path.exists("config.py"):
        # 检查是否已有备份
        if not os.path.exists("config.py.bak"):
            shutil.copy2("config.py", "config.py.bak")
            print("✅ 已创建配置文件备份: config.py.bak")
        return True
    else:
        print("❌ 未找到config.py文件")
        return False

def restore_config_backup():
    """恢复config.py备份"""
    if os.path.exists("config.py.bak"):
        shutil.copy2("config.py.bak", "config.py")
        print("✅ 已从备份恢复配置文件")
        return True
    else:
        print("❌ 未找到config.py.bak备份文件")
        return False

def get_current_config():
    """获取当前配置信息"""
    try:
        # 动态导入config模块
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        current_config = {
            "USE_ONNX": getattr(config, "USE_ONNX", False),
            "DEVICE": getattr(config, "DEVICE", "cpu"),
            "EP_LIST": getattr(config, "EP_LIST", []),
            "USE_MULTI_GPU": False,  # 默认值
            "PROPAINTER_MAX_LOAD_NUM": getattr(config, "PROPAINTER_MAX_LOAD_NUM", 10)
        }
        
        # 检查是否使用多GPU
        if hasattr(config, "DataParallel") or "DataParallel" in open("config.py").read():
            current_config["USE_MULTI_GPU"] = True
            
        return current_config
    except Exception as e:
        print(f"❌ 读取当前配置失败: {str(e)}")
        return None

def update_config_file(mode):
    """更新config.py文件以启用指定的GPU加速模式"""
    print_title(f"正在切换到 {mode} 模式")
    
    # 备份原始配置
    if not backup_config_file():
        return False
    
    try:
        with open("config.py", "r", encoding="utf-8") as f:
            config_content = f.read()
        
        # 根据模式更新配置内容
        if mode == "single_gpu":
            # 启用单GPU模式
            config_content = set_single_gpu_mode(config_content)
        elif mode == "multi_gpu":
            # 启用多GPU模式
            config_content = set_multi_gpu_mode(config_content)
        elif mode == "onnx":
            # 启用ONNX单GPU模式
            config_content = set_onnx_mode(config_content, multi_gpu=False)
        elif mode == "onnx_multi_gpu":
            # 启用ONNX多GPU模式
            config_content = set_onnx_mode(config_content, multi_gpu=True)
        else:
            print(f"❌ 不支持的模式: {mode}")
            return False
        
        # 写入更新后的配置
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"✅ 已成功切换到 {mode} 模式")
        return True
    
    except Exception as e:
        print(f"❌ 更新配置文件失败: {str(e)}")
        # 出错时恢复备份
        restore_config_backup()
        return False

def set_single_gpu_mode(config_content):
    """设置为单GPU模式"""
    # 禁用ONNX
    config_content = re.sub(r"USE_ONNX\s*=\s*True", "USE_ONNX = False", config_content)
    if "USE_ONNX" not in config_content:
        config_content += "\n# 禁用ONNX加速\nUSE_ONNX = False\n"
    
    # 设置为单GPU模式
    config_content = remove_dataparallel_code(config_content)
    
    # 设置DEVICE为cuda
    config_content = re.sub(r"DEVICE\s*=\s*['\"]cpu['\"]", "DEVICE = 'cuda'", config_content)
    if "DEVICE" not in config_content:
        config_content += "\n# 启用CUDA设备\nDEVICE = 'cuda'\n"
    
    return config_content

def set_multi_gpu_mode(config_content):
    """设置为多GPU模式"""
    # 禁用ONNX
    config_content = re.sub(r"USE_ONNX\s*=\s*True", "USE_ONNX = False", config_content)
    if "USE_ONNX" not in config_content:
        config_content += "\n# 禁用ONNX加速\nUSE_ONNX = False\n"
    
    # 设置DEVICE为cuda
    config_content = re.sub(r"DEVICE\s*=\s*['\"]cpu['\"]", "DEVICE = 'cuda'", config_content)
    if "DEVICE" not in config_content:
        config_content += "\n# 启用CUDA设备\nDEVICE = 'cuda'\n"
    
    # 添加DataParallel支持
    if "from torch.nn import DataParallel" not in config_content:
        config_content = "import torch\nfrom torch.nn import DataParallel\n" + config_content
    
    # 添加多GPU模型包装代码
    if "def wrap_model_multi_gpu" not in config_content:
        multi_gpu_code = """
# 多GPU支持
def wrap_model_multi_gpu(model):
    \"\"\"将模型包装为DataParallel以支持多GPU\"\"\"
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        return DataParallel(model)
    return model
"""
        config_content += multi_gpu_code
    
    return config_content

def set_onnx_mode(config_content, multi_gpu=False):
    """设置ONNX加速模式"""
    # 启用ONNX
    config_content = re.sub(r"USE_ONNX\s*=\s*False", "USE_ONNX = True", config_content)
    if "USE_ONNX" not in config_content:
        config_content += "\n# 启用ONNX加速\nUSE_ONNX = True\n"
    
    # 设置DEVICE为cuda
    config_content = re.sub(r"DEVICE\s*=\s*['\"]cpu['\"]", "DEVICE = 'cuda'", config_content)
    if "DEVICE" not in config_content:
        config_content += "\n# 启用CUDA设备\nDEVICE = 'cuda'\n"
    
    # 确保正确设置EP_LIST
    ep_list_code = """
# 设置ONNX执行提供商
EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']
"""
    if "EP_LIST" not in config_content:
        config_content += ep_list_code
    else:
        # 确保CUDAExecutionProvider在列表中且位于第一位
        config_content = re.sub(
            r"EP_LIST\s*=\s*\[(.*?)\]", 
            "EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']", 
            config_content,
            flags=re.DOTALL
        )
    
    # 处理多GPU设置
    if multi_gpu:
        # 添加DataParallel支持
        if "from torch.nn import DataParallel" not in config_content:
            config_content = "import torch\nfrom torch.nn import DataParallel\n" + config_content
        
        # 添加多GPU模型包装代码
        if "def wrap_model_multi_gpu" not in config_content:
            multi_gpu_code = """
# 多GPU支持
def wrap_model_multi_gpu(model):
    \"\"\"将模型包装为DataParallel以支持多GPU\"\"\"
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        return DataParallel(model)
    return model
"""
            config_content += multi_gpu_code
    else:
        # 移除DataParallel相关代码
        config_content = remove_dataparallel_code(config_content)
    
    return config_content

def remove_dataparallel_code(config_content):
    """移除DataParallel相关代码"""
    # 移除import语句
    config_content = re.sub(r"from torch\.nn import DataParallel\n?", "", config_content)
    
    # 移除wrap_model_multi_gpu函数
    config_content = re.sub(r"# 多GPU支持\ndef wrap_model_multi_gpu.*?return model\n", "", 
                            config_content, flags=re.DOTALL)
    
    return config_content

def describe_current_mode():
    """描述当前配置的加速模式"""
    config = get_current_config()
    if not config:
        return
    
    print_title("当前GPU加速模式")
    
    # 确定当前模式
    if config["USE_ONNX"] and config["USE_MULTI_GPU"]:
        mode = "onnx_multi_gpu"
        mode_name = "ONNX加速 + 多GPU并行"
    elif config["USE_ONNX"]:
        mode = "onnx"
        mode_name = "ONNX加速 (单GPU)"
    elif config["USE_MULTI_GPU"]:
        mode = "multi_gpu"
        mode_name = "多GPU并行"
    else:
        mode = "single_gpu"
        mode_name = "单GPU模式"
    
    print(f"当前模式: {mode_name} ({mode})")
    print("\n配置详情:")
    print(f"  ONNX加速: {'启用' if config['USE_ONNX'] else '禁用'}")
    print(f"  多GPU并行: {'启用' if config['USE_MULTI_GPU'] else '禁用'}")
    print(f"  设备: {config['DEVICE']}")
    if config["USE_ONNX"]:
        print(f"  ONNX提供商: {config['EP_LIST']}")
    print(f"  批处理大小: {config['PROPAINTER_MAX_LOAD_NUM']}")
    
    return mode

def interactive_mode_selection():
    """交互式模式选择"""
    print_title("GPU加速模式选择")
    
    print("可用的GPU加速模式:")
    print("1. 单GPU模式 (single_gpu): 稳定可靠的单GPU加速")
    print("2. 多GPU并行 (multi_gpu): 使用多个GPU并行处理")
    print("3. ONNX加速 (onnx): 单GPU下使用ONNX加速提高性能")
    print("4. ONNX加速+多GPU (onnx_multi_gpu): 最大性能模式")
    
    choice = input("\n请选择模式 (1-4): ")
    try:
        choice = int(choice)
        if choice == 1:
            return "single_gpu"
        elif choice == 2:
            return "multi_gpu"
        elif choice == 3:
            return "onnx"
        elif choice == 4:
            return "onnx_multi_gpu"
        else:
            print("❌ 无效选择，请输入1-4之间的数字")
            return None
    except ValueError:
        print("❌ 无效输入，请输入数字")
        return None

def main():
    parser = argparse.ArgumentParser(description='GPU加速模式切换工具')
    parser.add_argument('--mode', type=str, choices=['single_gpu', 'multi_gpu', 'onnx', 'onnx_multi_gpu'],
                        help='要切换到的GPU加速模式')
    parser.add_argument('--restore', action='store_true', help='从备份恢复配置文件')
    parser.add_argument('--info', action='store_true', help='显示当前GPU加速模式信息')
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("\n🚀 视频字幕移除工具 - GPU加速模式切换工具\n")
    
    # 恢复备份
    if args.restore:
        restore_config_backup()
        return
    
    # 显示当前模式信息
    current_mode = describe_current_mode()
    if args.info:
        return
    
    # 确定要切换的模式
    target_mode = args.mode
    if not target_mode:
        target_mode = interactive_mode_selection()
    
    # 如果选择的模式与当前模式相同，询问是否继续
    if target_mode and target_mode == current_mode:
        print(f"\n⚠️ 您选择的模式与当前模式相同: {target_mode}")
        confirm = input("是否继续切换? (y/n): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    # 执行模式切换
    if target_mode:
        update_config_file(target_mode)
        print("\n✅ 模式切换完成!")
        print("您可以运行 python check_gpu_usage.py 来验证新模式是否正确配置")

if __name__ == "__main__":
    main() 