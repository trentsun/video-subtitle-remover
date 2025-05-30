#!/bin/bash
# 视频字幕处理脚本 - 选择处理模式并开始处理

# 检查Python是否安装
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请安装Python 3.8或更高版本"
    exit 1
fi

# 显示当前模式
current_mode=$(python switch_mode.py status 2>/dev/null | grep "当前处理模式" | cut -d":" -f2 | xargs)

# 如果没有获取到模式信息，设置为未知
if [ -z "$current_mode" ]; then
    current_mode="未知"
fi

# 清屏并显示菜单
clear
echo "======================================================"
echo "            视频字幕去除工具 - 处理模式选择            "
echo "======================================================"
echo ""
echo "当前处理模式: $current_mode"
echo ""
echo "请选择视频处理模式:"
echo ""
echo "  1) 单帧处理模式 - 稳定但速度较慢，适合所有硬件"
echo "  2) 批处理模式 - 速度快但需要更多显存，可能出现OOM错误"
echo "  3) 查看显存状态"
echo "  4) 直接使用当前模式处理视频"
echo "  0) 退出"
echo ""
echo "提示: 如果视频处理过程中出现卡顿或者内存错误，请使用单帧处理模式"
echo ""
read -p "请输入选项 [0-4]: " choice

case $choice in
    1)
        echo "切换到单帧处理模式..."
        python switch_mode.py single
        ;;
    2)
        echo ""
        read -p "请输入批处理大小 (推荐: 8-32，根据显存大小调整): " batch_size
        if [[ ! $batch_size =~ ^[0-9]+$ ]] || [ $batch_size -lt 2 ]; then
            echo "批处理大小必须是大于1的整数，使用默认值16"
            batch_size=16
        fi
        echo "切换到批处理模式，批处理大小: $batch_size..."
        python switch_mode.py batch --batch-size $batch_size
        ;;
    3)
        echo "正在获取GPU显存状态..."
        echo ""
        python check_memory.py -i 3 -d 10
        # 显示完成后返回到菜单
        exec $0
        exit 0
        ;;
    4)
        # 继续处理
        echo "使用当前模式 ($current_mode) 处理视频..."
        ;;
    0)
        echo "退出程序"
        exit 0
        ;;
    *)
        echo "无效选项，使用当前模式处理视频..."
        ;;
esac

# 设置批处理大小环境变量
if [[ $current_mode == *"批处理"* ]]; then
    batch_size=$(echo $current_mode | grep -o '[0-9]\+' || echo "16")
    export MAX_BATCH_SIZE=$batch_size
else
    export MAX_BATCH_SIZE=1
fi

# 环境变量设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,backend:cudaMallocAsync"
export CUDA_VISIBLE_DEVICES=0  # 使用单GPU模式
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO

# 启用同步模式以便于定位问题
if [[ $current_mode == *"单帧"* ]]; then
    export CUDA_LAUNCH_BLOCKING=1
else
    export CUDA_LAUNCH_BLOCKING=0
fi

echo ""
echo "准备处理视频..."
echo "处理模式: $current_mode"
echo "最大批处理大小: $MAX_BATCH_SIZE"
echo ""

# 使用超时保护
TIMEOUT=1800 # 30分钟超时
timeout $TIMEOUT python backend/main.py "$@" || { 
    echo "程序执行超时或错误，退出代码: $?"
    # 清理所有可能残留的进程
    pkill -f "python backend/main.py"
    exit 1
}

echo "程序执行完成" 