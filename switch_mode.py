#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
切换处理模式工具 - 在单帧处理和批处理模式之间切换
"""

import os
import sys
import re
import argparse
import shutil
from datetime import datetime

CONFIG_FILE = "backend/config.py"
BACKUP_DIR = "config_backups"

def create_backup(config_file):
    """创建配置文件的备份"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"config_{timestamp}.py")
    
    shutil.copy2(config_file, backup_file)
    print(f"已创建配置文件备份: {backup_file}")
    return backup_file

def set_single_frame_mode(enable=True):
    """设置单帧处理模式"""
    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 未找到配置文件 {CONFIG_FILE}")
        return False
    
    # 先创建备份
    backup_file = create_backup(CONFIG_FILE)
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改批处理大小
    if enable:
        batch_size = 1
        content = re.sub(r'PROPAINTER_MAX_LOAD_NUM\s*=\s*\d+.*', 
                        f'PROPAINTER_MAX_LOAD_NUM = {batch_size}  # 设置为1启用单帧处理模式', 
                        content)
    else:
        batch_size = 16
        content = re.sub(r'PROPAINTER_MAX_LOAD_NUM\s*=\s*\d+.*', 
                        f'PROPAINTER_MAX_LOAD_NUM = {batch_size}  # 使用批处理模式', 
                        content)
    
    # 修改强制单帧处理模式选项
    if enable:
        content = re.sub(r'PROPAINTER_FORCE_SINGLE_FRAME\s*=\s*(?:True|False).*', 
                        'PROPAINTER_FORCE_SINGLE_FRAME = True  # 启用: 强制使用单帧处理', 
                        content)
    else:
        content = re.sub(r'PROPAINTER_FORCE_SINGLE_FRAME\s*=\s*(?:True|False).*', 
                        'PROPAINTER_FORCE_SINGLE_FRAME = False  # 禁用: 允许使用批处理', 
                        content)
    
    # 如果找不到PROPAINTER_FORCE_SINGLE_FRAME选项，添加它
    if 'PROPAINTER_FORCE_SINGLE_FRAME' not in content:
        insert_point = content.find('# ×××××××××× InpaintMode.PROPAINTER算法设置 end ××××××××××')
        if insert_point > 0:
            force_single_line = f'# 是否强制使用单帧处理模式，不管批处理设置如何\nPROPAINTER_FORCE_SINGLE_FRAME = {"True" if enable else "False"}  # {"启用" if enable else "禁用"}: {"强制使用单帧处理" if enable else "允许使用批处理"}\n'
            content = content[:insert_point] + force_single_line + content[insert_point:]
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    mode_name = "单帧处理" if enable else "批处理"
    print(f"已成功切换到{mode_name}模式!")
    if enable:
        print("现在系统将使用单帧处理，处理速度较慢但稳定，适合所有硬件配置")
    else:
        print("现在系统将使用批处理模式，处理速度更快但需要更多显存，如果出现OOM错误请切换回单帧模式")
    
    return True

def get_current_mode():
    """获取当前的处理模式"""
    if not os.path.exists(CONFIG_FILE):
        return "未知"
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否强制单帧模式
    force_single_match = re.search(r'PROPAINTER_FORCE_SINGLE_FRAME\s*=\s*(True|False)', content)
    if force_single_match and force_single_match.group(1) == 'True':
        return "单帧处理"
    
    # 检查批处理大小
    batch_size_match = re.search(r'PROPAINTER_MAX_LOAD_NUM\s*=\s*(\d+)', content)
    if batch_size_match:
        batch_size = int(batch_size_match.group(1))
        if batch_size <= 1:
            return "单帧处理"
        else:
            return f"批处理 (大小: {batch_size})"
    
    return "未知"

def main():
    parser = argparse.ArgumentParser(description='切换视频处理模式 (单帧/批处理)')
    parser.add_argument('mode', choices=['single', 'batch', 'status'], 
                        help='切换模式: single=单帧处理, batch=批处理, status=查看当前状态')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批处理模式的批处理大小 (仅在使用batch模式时有效)')
    
    args = parser.parse_args()
    
    if args.mode == 'status':
        current_mode = get_current_mode()
        print(f"当前处理模式: {current_mode}")
        return
    
    # 切换到单帧处理模式
    if args.mode == 'single':
        set_single_frame_mode(True)
    
    # 切换到批处理模式
    elif args.mode == 'batch':
        set_single_frame_mode(False)
        
        # 如果指定了批处理大小，修改它
        if args.batch_size > 1:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = re.sub(r'PROPAINTER_MAX_LOAD_NUM\s*=\s*\d+.*', 
                            f'PROPAINTER_MAX_LOAD_NUM = {args.batch_size}  # 使用批处理模式', 
                            content)
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"批处理大小设置为: {args.batch_size}")

if __name__ == '__main__':
    main() 