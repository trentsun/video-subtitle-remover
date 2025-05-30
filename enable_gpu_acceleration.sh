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
    echo_title "检查Python环境"
    
    if command -v python3 &>/dev/null; then
        PY_CMD="python3"
    elif command -v python &>/dev/null; then
        PY_CMD="python"
    else
        echo_error "未找到Python！请先安装Python 3.6+"
        exit 1
    fi
    
    PY_VERSION=$($PY_CMD --version)
    echo_success "找到Python: $PY_VERSION"
    
    # 检查pip
    if command -v pip3 &>/dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &>/dev/null; then
        PIP_CMD="pip"
    else
        echo_error "未找到pip！请先安装pip"
        exit 1
    fi
    
    echo_success "找到pip: $($PIP_CMD --version)"
}

# 检查CUDA
check_cuda() {
    echo_title "检查CUDA环境"
    
    if command -v nvidia-smi &>/dev/null; then
        echo_success "找到NVIDIA驱动"
        nvidia-smi
    else
        echo_error "未找到NVIDIA驱动，GPU加速可能无法正常工作！"
        echo_warning "继续安装，但GPU加速可能不可用"
    fi
}

# 安装依赖
install_dependencies() {
    echo_title "安装GPU加速相关依赖"
    
    echo_info "正在安装onnxruntime-gpu..."
    $PIP_CMD install onnxruntime-gpu --upgrade
    
    echo_info "正在安装其他GPU加速依赖..."
    $PIP_CMD install torch torchvision --upgrade
    
    echo_success "依赖安装完成"
}

# 运行修复脚本
run_fix_scripts() {
    echo_title "运行修复脚本"
    
    echo_info "正在修复配置问题..."
    $PY_CMD fix_config.py --all
    
    echo_success "修复脚本执行完成"
}

# 设置GPU加速模式
setup_gpu_mode() {
    echo_title "设置GPU加速模式"
    
    echo_info "根据系统检测选择合适的加速模式..."
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    
    if [ "$GPU_COUNT" -gt "1" ]; then
        echo_info "检测到 $GPU_COUNT 个GPU，推荐使用 ONNX+多GPU 模式"
        $PY_CMD toggle_gpu_mode.py --mode onnx_multi_gpu
    else
        echo_info "检测到 $GPU_COUNT 个GPU，推荐使用 ONNX单GPU 模式"
        $PY_CMD toggle_gpu_mode.py --mode onnx
    fi
    
    echo_success "GPU加速模式设置完成"
}

# 验证GPU加速
validate_gpu_acceleration() {
    echo_title "验证GPU加速"
    
    $PY_CMD check_gpu_usage.py
    
    echo_info "如上方显示，您可以检查GPU加速是否正确配置"
    echo_info "如果要进行更详细的测试，请使用: python check_gpu_usage.py --video your_video.mp4"
}

# 主函数
main() {
    echo_title "视频字幕去除工具 - GPU加速设置向导"
    
    # 检查环境
    check_python
    check_cuda
    
    # 询问是否安装依赖
    echo -e "\n是否安装GPU加速所需的依赖? (y/n)"
    read -r install_deps
    if [[ $install_deps =~ ^[Yy]$ ]]; then
        install_dependencies
    else
        echo_info "跳过依赖安装"
    fi
    
    # 运行修复脚本
    echo -e "\n是否修复配置问题? (y/n)"
    read -r run_fixes
    if [[ $run_fixes =~ ^[Yy]$ ]]; then
        run_fix_scripts
    else
        echo_info "跳过配置修复"
    fi
    
    # 设置GPU加速模式
    echo -e "\n是否设置GPU加速模式? (y/n)"
    read -r setup_mode
    if [[ $setup_mode =~ ^[Yy]$ ]]; then
        setup_gpu_mode
    else
        echo_info "跳过GPU加速模式设置"
    fi
    
    # 验证GPU加速
    echo -e "\n是否验证GPU加速设置? (y/n)"
    read -r validate
    if [[ $validate =~ ^[Yy]$ ]]; then
        validate_gpu_acceleration
    else
        echo_info "跳过GPU加速验证"
    fi
    
    echo_title "GPU加速设置完成"
    echo_success "所有操作已完成！"
    echo_info "如果配置没有问题，你的视频处理应该能够使用GPU加速了"
    echo_info "有任何问题，请查看 'GPU加速问题修复说明.md' 获取详细信息"
}

# 执行主函数
main 