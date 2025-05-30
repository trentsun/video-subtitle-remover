#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import shutil
import importlib.util
import argparse

def check_config_file():
    """检查config.py是否存在"""
    if not os.path.exists('config.py'):
        print("❌ 未找到config.py文件")
        return False
    return True

def backup_config_file():
    """备份config.py文件"""
    if os.path.exists('config.py'):
        backup_path = 'config.py.backup'
        shutil.copy2('config.py', backup_path)
        print(f"✅ 已创建配置文件备份: {backup_path}")
        return True
    return False

def fix_onnx_providers():
    """修复ONNX提供商配置"""
    print("\n正在修复ONNX提供商配置...")
    if not check_config_file():
        return False
    
    # 备份配置文件
    backup_config_file()
    
    try:
        # 读取配置文件
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有USE_ONNX定义
        use_onnx_match = re.search(r'USE_ONNX\s*=\s*(True|False)', content)
        
        # 检查是否有EP_LIST定义
        ep_list_match = re.search(r'EP_LIST\s*=\s*\[(.*?)\]', content, re.DOTALL)
        
        # 新的配置内容
        new_config = """
# ONNX加速配置
USE_ONNX = True

# ONNX执行提供商列表
EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']
"""
        
        # 如果没有找到定义，添加新的定义
        if not use_onnx_match and not ep_list_match:
            # 找到合适的位置添加配置
            insert_point = content.find("# 配置")
            if insert_point < 0:
                insert_point = 0
            
            # 添加到合适位置
            content = content[:insert_point] + new_config + content[insert_point:]
            print("✅ 已添加ONNX配置")
        else:
            # 替换现有的定义
            if use_onnx_match:
                content = re.sub(r'USE_ONNX\s*=\s*False', 'USE_ONNX = True', content)
                print("✅ 已启用USE_ONNX")
            
            if ep_list_match:
                content = re.sub(
                    r'EP_LIST\s*=\s*\[(.*?)\]', 
                    "EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']", 
                    content, 
                    flags=re.DOTALL
                )
                print("✅ 已设置CUDA执行提供商")
        
        # 写回配置文件
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ ONNX提供商配置修复完成")
        return True
    
    except Exception as e:
        print(f"❌ 修复ONNX提供商配置失败: {str(e)}")
        return False

def fix_dataparallel_import():
    """修复DataParallel导入问题"""
    print("\n正在修复DataParallel导入问题...")
    if not check_config_file():
        return False
    
    try:
        # 读取配置文件
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有DataParallel导入
        dataparallel_import = "from torch.nn import DataParallel" in content
        
        # 如果没有DataParallel导入，添加导入语句
        if not dataparallel_import:
            # 添加导入语句到文件开头
            import_line = "import torch\nfrom torch.nn import DataParallel\n"
            content = import_line + content
            print("✅ 已添加DataParallel导入")
            
            # 添加wrap_model_multi_gpu函数
            wrap_function = """
# 多GPU支持
def wrap_model_multi_gpu(model):
    \"\"\"将模型包装为DataParallel以支持多GPU\"\"\"
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        return DataParallel(model)
    return model
"""
            # 添加到文件末尾
            content += wrap_function
            print("✅ 已添加wrap_model_multi_gpu函数")
        
        # 写回配置文件
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ DataParallel导入问题修复完成")
        return True
    
    except Exception as e:
        print(f"❌ 修复DataParallel导入问题失败: {str(e)}")
        return False

def fix_device_setting():
    """修复DEVICE设置"""
    print("\n正在修复DEVICE设置...")
    if not check_config_file():
        return False
    
    try:
        # 读取配置文件
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查DEVICE设置
        device_match = re.search(r'DEVICE\s*=\s*[\'"](\w+)[\'"]', content)
        
        if device_match:
            # 如果DEVICE已定义但不是cuda，则替换为cuda
            if device_match.group(1) != 'cuda':
                content = re.sub(r'DEVICE\s*=\s*[\'"](\w+)[\'"]', "DEVICE = 'cuda'", content)
                print("✅ 已将DEVICE设置更新为cuda")
        else:
            # 如果未定义DEVICE，添加定义
            device_code = "\n# 设置默认设备为CUDA\nDEVICE = 'cuda'\n"
            # 找个合适的位置添加
            use_onnx_position = content.find("USE_ONNX")
            if use_onnx_position > 0:
                # 添加到USE_ONNX的后面
                end_line = content.find("\n", use_onnx_position)
                content = content[:end_line+1] + device_code + content[end_line+1:]
            else:
                # 添加到文件开头
                content = device_code + content
            print("✅ 已添加DEVICE设置")
        
        # 写回配置文件
        with open('config.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ DEVICE设置修复完成")
        return True
    
    except Exception as e:
        print(f"❌ 修复DEVICE设置失败: {str(e)}")
        return False

def modify_detector_file():
    """修改sub_detector.py文件以使用GPU和ONNX加速"""
    print("\n正在修改字幕检测器以启用GPU加速...")
    detector_file = 'sub_detector.py'
    if not os.path.exists(detector_file):
        print(f"❌ 未找到{detector_file}文件")
        return False
    
    # 备份配置文件
    backup_path = f'{detector_file}.backup'
    shutil.copy2(detector_file, backup_path)
    print(f"✅ 已创建检测器文件备份: {backup_path}")
    
    try:
        # 读取配置文件
        with open(detector_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 寻找text_detector方法
        text_detector_match = re.search(r'def text_detector\(self, frame\):', content)
        
        if text_detector_match:
            # 替换text_detector方法中的关键部分
            detector_pattern = r'(def text_detector\(self, frame\):.*?return dt_boxes\s*\n)'
            # 使用非贪婪匹配和DOTALL标志
            detector_method = re.search(detector_pattern, content, re.DOTALL)
            
            if detector_method:
                old_method = detector_method.group(1)
                
                # 修改后的方法，添加GPU支持
                new_method = """def text_detector(self, frame):
        \"\"\"
        文本检测
        \"\"\"
        import sys
        import os
        from config import USE_ONNX, DEVICE
        
        if USE_ONNX:
            try:
                # 使用ONNX加速文本检测
                from config import EP_LIST
                dt_boxes = self.text_detector_onnx(frame)
                return dt_boxes
            except Exception as e:
                print(f"ONNX加速失败，回退到标准模式: {e}")
                pass
                
        # 标准检测逻辑
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像输入到处理管道中
        dt_boxes, _ = self.text_handle.process(img_rgb, rec=False)
        
        # 确保在GPU上运行
        dt_boxes = np.array(dt_boxes)
        return dt_boxes
        
"""
                # 替换方法
                content = content.replace(old_method, new_method)
                print("✅ 已更新text_detector方法以支持GPU和ONNX")
            else:
                print("❌ 无法定位text_detector方法的完整实现")
        else:
            print("❌ 未找到text_detector方法")
        
        # 检查是否有text_detector_onnx方法，如果没有则添加
        if "def text_detector_onnx" not in content:
            # 添加ONNX检测方法
            onnx_method = """
    def text_detector_onnx(self, frame):
        \"\"\"
        使用ONNX Runtime加速的文本检测
        \"\"\"
        import onnxruntime as ort
        from config import EP_LIST
        import numpy as np
        
        try:
            # 检查ONNX模型是否存在，如果不存在则创建
            onnx_model_path = os.path.join(os.path.dirname(__file__), "models", "text_detector.onnx")
            if not os.path.exists(onnx_model_path):
                print("ONNX模型不存在，正在创建...")
                self.export_to_onnx(onnx_model_path)
            
            # 创建ONNX会话
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=EP_LIST)
            
            # 预处理图像
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取输入名称
            input_name = session.get_inputs()[0].name
            
            # 运行推理
            result = session.run(None, {input_name: np.array([img_rgb], dtype=np.float32)})[0]
            
            # 后处理结果
            dt_boxes = result[0]  # 假设第一个输出是检测框
            return dt_boxes
            
        except Exception as e:
            print(f"ONNX推理失败: {e}")
            # 回退到标准模式
            return self.text_detector_standard(frame)
    
    def text_detector_standard(self, frame):
        \"\"\"
        标准模式的文本检测（回退方法）
        \"\"\"
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dt_boxes, _ = self.text_handle.process(img_rgb, rec=False)
        return np.array(dt_boxes)
    
    def export_to_onnx(self, output_path):
        \"\"\"
        将文本检测模型导出为ONNX格式
        \"\"\"
        try:
            import torch
            import torch.onnx
            from pathlib import Path
            
            # 创建输出目录
            Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
            
            # 获取模型
            model = self.text_handle.model  # 假设可以这样访问模型
            
            # 创建示例输入
            dummy_input = torch.randn(1, 3, 640, 640, device="cuda")  # 调整大小为模型实际输入
            
            # 导出为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
            
            print(f"ONNX模型已成功导出到: {output_path}")
            return True
        except Exception as e:
            print(f"导出ONNX模型失败: {e}")
            return False
"""
            # 将方法添加到类中合适的位置
            class_end = content.rfind("}")
            if class_end > 0:
                content = content[:class_end] + onnx_method + content[class_end:]
                print("✅ 已添加ONNX加速检测方法")
            else:
                # 添加到文件末尾
                content += "\n" + onnx_method
                print("✅ 已添加ONNX加速检测方法（到文件末尾）")
        else:
            print("✅ 文件中已存在text_detector_onnx方法")
        
        # 写回文件
        with open(detector_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 字幕检测器修改完成")
        return True
    
    except Exception as e:
        print(f"❌ 修改字幕检测器失败: {str(e)}")
        # 恢复备份
        shutil.copy2(backup_path, detector_file)
        print(f"✅ 已从备份恢复检测器文件")
        return False

def main():
    parser = argparse.ArgumentParser(description='自动修复字幕检测器的GPU加速问题')
    parser.add_argument('--onnx', action='store_true', help='修复ONNX提供商配置')
    parser.add_argument('--device', action='store_true', help='修复DEVICE设置')
    parser.add_argument('--dataparallel', action='store_true', help='修复DataParallel导入问题')
    parser.add_argument('--detector', action='store_true', help='修改字幕检测器以启用GPU加速')
    parser.add_argument('--all', action='store_true', help='执行所有修复')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  字幕检测器GPU加速问题自动修复工具")
    print("=" * 80)
    
    # 如果没有指定具体选项，默认执行全部
    if not (args.onnx or args.device or args.dataparallel or args.detector):
        args.all = True
    
    # 执行选定的修复
    if args.all or args.onnx:
        fix_onnx_providers()
    
    if args.all or args.device:
        fix_device_setting()
    
    if args.all or args.dataparallel:
        fix_dataparallel_import()
    
    if args.all or args.detector:
        modify_detector_file()
    
    print("\n✅ 修复操作完成!")
    print("建议运行 python check_gpu_usage.py 检查修复效果")

if __name__ == "__main__":
    main() 