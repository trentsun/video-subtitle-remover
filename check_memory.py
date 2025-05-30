#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU内存监控工具，帮助诊断内存问题
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import argparse

def get_gpu_memory_info():
    """获取GPU内存使用信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                 '--format=csv,noheader,nounits'], 
                                check=True, stdout=subprocess.PIPE, universal_newlines=True)
        return result.stdout.strip().split('\n')
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def monitor_memory(interval=1.0, log_file=None, duration=None):
    """监控GPU内存使用情况
    
    Args:
        interval: 刷新间隔（秒）
        log_file: 日志文件路径
        duration: 监控持续时间（秒）
    """
    if log_file:
        f = open(log_file, 'w')
        f.write("时间,GPU ID,GPU名称,已用内存(MB),总内存(MB),GPU利用率(%)\n")
    else:
        f = None
    
    start_time = time.time()
    try:
        while True:
            gpu_info = get_gpu_memory_info()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"===== GPU内存监控 [{current_time}] =====")
            if gpu_info:
                print("GPU ID  |  名称  |  已用内存/总内存  |  利用率")
                print("-" * 60)
                
                for line in gpu_info:
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_id, name, used, total, util = parts
                        print(f"{gpu_id:^7} | {name[:10]:^10} | {used:>6}/{total:<6} MB | {util:>5}%")
                        
                        if f:
                            f.write(f"{current_time},{gpu_id},{name},{used},{total},{util}\n")
                            f.flush()
            else:
                print("未找到NVIDIA GPU或nvidia-smi命令不可用")
                
                if f:
                    f.write(f"{current_time},无GPU信息\n")
                    f.flush()
            
            # 如果设置了持续时间，检查是否已超过
            if duration and (time.time() - start_time >= duration):
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
    finally:
        if f:
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='监控GPU内存使用情况')
    parser.add_argument('-i', '--interval', type=float, default=1.0, help='刷新间隔（秒）')
    parser.add_argument('-l', '--log', type=str, help='日志文件路径')
    parser.add_argument('-d', '--duration', type=int, help='监控持续时间（秒）')
    
    args = parser.parse_args()
    
    print("GPU内存监控工具")
    print("按Ctrl+C停止监控")
    print("")
    
    monitor_memory(interval=args.interval, log_file=args.log, duration=args.duration) 