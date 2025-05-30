#!/bin/bash
# 设置PyTorch显存分配策略，避免内存碎片化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,backend:cudaMallocAsync"

# 限制每个进程使用的GPU数量，避免一个进程占用所有GPU
# 根据您的实际情况，如果有多个GPU，可以尝试只使用单GPU或特定GPU
# export CUDA_VISIBLE_DEVICES=0  # 仅使用第一个GPU
export CUDA_VISIBLE_DEVICES=0,1  # 使用前两个GPU

# 设置批处理大小的环境变量，以便在不修改配置文件的情况下动态调整
export MAX_BATCH_SIZE=16  # 提高批处理大小以加快处理速度

# 设置PyTorch并行度参数，提高多GPU利用率
export OMP_NUM_THREADS=8  # 增加到8
export MKL_NUM_THREADS=8  # 增加到8

# 设置GPU操作的异步执行，提高性能
export CUDA_LAUNCH_BLOCKING=0  # 改为0启用异步执行，提高性能

# 减少PyTorch内存缓存限制
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # 注释掉以允许内存缓存复用

# 启用NVIDIA GPU优化
export NVIDIA_TF32_OVERRIDE=1  # 启用TF32优化

# 开启PyTorch异常详细信息
export TORCH_SHOW_CPP_STACKTRACES=1

# 启用PyTorch详细日志
export TORCH_CPP_LOG_LEVEL=INFO

echo "环境变量设置完成，启动程序..."
echo "当前设置批处理大小为: $MAX_BATCH_SIZE"
echo "如需更大批处理大小或遇到问题，请修改 config.py 中的设置"

# 添加超时保护
(sleep 1800 && echo "程序运行超过30分钟，可能已卡住，自动终止..." && pkill -f "python backend/main.py") &
timeout_pid=$!

# 启动主程序
python backend/main.py "$@"
result=$?

# 清理超时进程
kill $timeout_pid 2>/dev/null

exit $result 