#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的标题
echo_title() {
    echo -e "\n${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

# 输出成功消息
echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 输出警告消息
echo_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# 输出错误消息
echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 输出信息
echo_info() {
    echo -e "  $1"
}

# 检查Python环境
check_python() {
    if command -v python3 &>/dev/null; then
        PY_CMD="python3"
    elif command -v python &>/dev/null; then
        PY_CMD="python"
    else
        echo_error "未找到Python！请先安装Python 3.6+"
        exit 1
    fi
}

# 检查GPU状态
check_gpu() {
    echo_title "GPU状态检查"
    
    if command -v nvidia-smi &>/dev/null; then
        echo_success "NVIDIA GPU可用"
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
        GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        echo_info "可见GPU: $CUDA_VISIBLE_DEVICES"
        echo_info "GPU数量: $GPU_COUNT"
        
        # 显示显存使用情况
        echo_info "GPU显存使用情况:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    else
        echo_error "未找到NVIDIA驱动，GPU加速不可用！"
        echo_warning "将以CPU模式运行，处理速度可能较慢"
    fi
}

# 检查模型是否已转换为ONNX
check_onnx_models() {
    echo_title "ONNX模型检查"
    
    MODELS_DIR="models"
    if [ ! -d "$MODELS_DIR" ]; then
        echo_warning "未找到models目录，将在首次运行时自动创建ONNX模型"
        return
    fi
    
    # 检查是否存在ONNX模型
    ONNX_MODELS=$(find "$MODELS_DIR" -name "*.onnx" | wc -l)
    if [ "$ONNX_MODELS" -gt "0" ]; then
        echo_success "找到 $ONNX_MODELS 个ONNX模型"
        find "$MODELS_DIR" -name "*.onnx" | while read -r model; do
            echo_info "- $model"
        done
    else
        echo_warning "未找到ONNX模型，将在首次运行时自动创建"
    fi
}

# 检查配置是否为GPU加速模式
check_gpu_config() {
    echo_title "GPU配置检查"
    
    # 使用python脚本检查配置
    $PY_CMD -c "
import os
try:
    from config import USE_ONNX, DEVICE
    print(f'ONNX加速: {'启用' if USE_ONNX else '禁用'}')
    print(f'设备: {DEVICE}')
    if 'cuda' in DEVICE:
        print('✅ 配置为GPU加速模式')
    else:
        print('❌ 未配置为GPU加速模式')
        print('建议运行: ./enable_gpu_acceleration.sh')
except ImportError as e:
    print(f'❌ 配置文件读取错误: {e}')
    print('建议运行: ./enable_gpu_acceleration.sh')
except Exception as e:
    print(f'❌ 配置检查错误: {e}')
"
}

# 运行字幕去除工具
run_tool() {
    echo_title "运行字幕去除工具 (GPU加速模式)"
    
    # 获取输入参数
    INPUT_VIDEO="$1"
    OUTPUT_VIDEO="${2:-output.mp4}"
    
    if [ -z "$INPUT_VIDEO" ]; then
        echo_error "未指定输入视频！"
        echo_info "用法: $0 <输入视频路径> [输出视频路径]"
        exit 1
    fi
    
    if [ ! -f "$INPUT_VIDEO" ]; then
        echo_error "输入视频文件不存在: $INPUT_VIDEO"
        exit 1
    fi
    
    echo_info "输入视频: $INPUT_VIDEO"
    echo_info "输出视频: $OUTPUT_VIDEO"
    
    # 设置GPU环境变量
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    
    # 运行主脚本
    echo_info "正在处理视频，请稍候..."
    start_time=$(date +%s)
    
    # 调用主程序
    $PY_CMD main.py --video "$INPUT_VIDEO" --output "$OUTPUT_VIDEO"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo_success "处理完成！用时: ${duration}秒"
    echo_info "输出视频保存在: $OUTPUT_VIDEO"
}

# 主函数
main() {
    echo_title "视频字幕去除工具 - GPU加速模式"
    
    # 检查环境
    check_python
    check_gpu
    check_onnx_models
    check_gpu_config
    
    # 检查是否需要设置GPU加速
    if ! $PY_CMD -c "
try:
    from config import USE_ONNX, DEVICE
    if USE_ONNX and 'cuda' in DEVICE:
        exit(0)
    else:
        exit(1)
except:
    exit(1)
"; then
        echo_warning "未启用GPU加速模式，是否运行GPU加速设置向导? (y/n)"
        read -r setup_gpu
        if [[ $setup_gpu =~ ^[Yy]$ ]]; then
            if [ -f "./enable_gpu_acceleration.sh" ]; then
                chmod +x ./enable_gpu_acceleration.sh
                ./enable_gpu_acceleration.sh
            else
                echo_error "未找到GPU加速设置向导脚本！"
                echo_info "请先确保 enable_gpu_acceleration.sh 存在"
            fi
        else
            echo_warning "跳过GPU加速设置，可能无法获得最佳性能"
        fi
    fi
    
    # 运行工具
    run_tool "$@"
}

# 执行主函数，传递所有命令行参数
main "$@" 