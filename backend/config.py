import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import onnxruntime as ort

# 项目版本号
VERSION = "1.1.1"
# ×××××××××××××××××××× [不要改] start ××××××××××××××××××××
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    USE_DML = False
    # 检测是否存在多个GPU并使用DataParallel
    if torch.cuda.device_count() > 1:
        # 使用所有可用GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        USE_MULTI_GPU = True
        AVAILABLE_GPUS = torch.cuda.device_count()
        print(f"发现 {AVAILABLE_GPUS} 个GPU设备，将启用多GPU加速")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        USE_MULTI_GPU = False
        AVAILABLE_GPUS = 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')

# 查看该路径下是否有模型完整文件，没有的话合并小文件生成完整文件
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# 指定ffmpeg可执行程序路径
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# 将ffmpeg添加可执行权限
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 是否使用ONNX加速
USE_ONNX = True

# ONNX执行提供商设置
ONNX_PROVIDERS = []
available_providers = ort.get_available_providers()
# 首先尝试添加GPU相关提供商
if "CUDAExecutionProvider" in available_providers:
    ONNX_PROVIDERS.append("CUDAExecutionProvider")
elif "DmlExecutionProvider" in available_providers:
    ONNX_PROVIDERS.append("DmlExecutionProvider")
elif "ROCMExecutionProvider" in available_providers:
    ONNX_PROVIDERS.append("ROCMExecutionProvider")
elif "MetalExecutionProvider" in available_providers:
    ONNX_PROVIDERS.append("MetalExecutionProvider")

# 如果没有GPU提供商，最后添加CPU作为后备
if not ONNX_PROVIDERS:
    ONNX_PROVIDERS.append("CPUExecutionProvider")

# 打印当前使用的ONNX执行提供商
print(f"ONNX执行提供商: {ONNX_PROVIDERS}")

# ONNX加速优化级别 (0-3)
# 0: 禁用优化
# 1: 基本优化
# 2: 扩展优化
# 3: 最大优化
ONNX_OPTIMIZATION_LEVEL = 3

@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ×××××××××××××××××××× [可以改] start ××××××××××××××××××××
# 是否使用h264编码，如果需要安卓手机分享生成的视频，请打开该选项
USE_H264 = True

# ×××××××××× 通用设置 start ××××××××××
"""
MODE可选算法类型
- InpaintMode.STTN 算法：对于真人视频效果较好，速度快，可以跳过字幕检测
- InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
- InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
"""
# 【设置inpaint算法】
MODE = InpaintMode.PROPAINTER
# 【设置像素点偏差】
# 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
SUBTITLE_AREA_DEVIATION_PIXEL = 60  # 进一步增大偏差像素值，从40调整到60
# 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
THRESHOLD_HEIGHT_DIFFERENCE = 25  # 略微提高高度差异容忍度
# 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
PIXEL_TOLERANCE_Y = 25  # 允许检测框纵向偏差的像素点数，提高到25
PIXEL_TOLERANCE_X = 25  # 允许检测框横向偏差的像素点数，提高到25

# ×××××××××× 字幕检测GPU加速设置 start ××××××××××
# 是否启用字幕检测批处理加速
SUBTITLE_BATCH_PROCESSING = True
# 最大批处理大小（为0则自动根据GPU显存调整）
# 较大的批处理大小可提高处理速度，但需要更多显存
SUBTITLE_MAX_BATCH_SIZE = 0
# 多线程数量（为0则自动根据CPU核心数调整）
# 较多的线程数可以提高处理速度，但可能增加系统负担
SUBTITLE_MAX_THREADS = 0
# 是否使用半精度(FP16)推理以节省显存
SUBTITLE_USE_FP16 = False
# 是否启用字幕检测结果缓存
ENABLE_SUBTITLE_CACHE = True
# 字幕检测结果缓存目录
SUBTITLE_CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR), '.subtitle_cache')
# 缓存文件名前缀
SUBTITLE_CACHE_PREFIX = 'subtitle_detection_'
# 是否在视频文件发生更改时强制重新检测
FORCE_REDETECT_ON_MODIFIED = True
# ×××××××××× 字幕检测GPU加速设置 end ××××××××××
# ×××××××××× 通用设置 end ××××××××××

# ×××××××××× InpaintMode.STTN算法设置 start ××××××××××
# 以下参数仅适用STTN算法时，才生效
"""
1. STTN_SKIP_DETECTION
含义：是否使用跳过检测
效果：设置为True跳过字幕检测，会省去很大时间，但是可能误伤无字幕的视频帧或者会导致去除的字幕漏了

2. STTN_NEIGHBOR_STRIDE
含义：相邻帧数步长, 如果需要为第50帧填充缺失的区域，STTN_NEIGHBOR_STRIDE=5，那么算法会使用第45帧、第40帧等作为参照。
效果：用于控制参考帧选择的密度，较大的步长意味着使用更少、更分散的参考帧，较小的步长意味着使用更多、更集中的参考帧。

3. STTN_REFERENCE_LENGTH
含义：参数帧数量，STTN算法会查看每个待修复帧的前后若干帧来获得用于修复的上下文信息
效果：调大会增加显存占用，处理效果变好，但是处理速度变慢

4. STTN_MAX_LOAD_NUM
含义：STTN算法每次最多加载的视频帧数量
效果：设置越大速度越慢，但效果越好
注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = False
# 参考帧步长
STTN_NEIGHBOR_STRIDE = 5  # 从10减小到5，使用更多更密集的参考帧
# 参考帧长度（数量）
STTN_REFERENCE_LENGTH = 40  # 从15增加到25，增加参考帧数量提高修复效果
# 设置STTN算法最大同时处理的帧数量
STTN_MAX_LOAD_NUM = 200  # 增加到200确保足够容纳参考帧
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN算法设置 end ××××××××××

# ×××××××××× InpaintMode.PROPAINTER算法设置 start ××××××××××
# 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
# 1280x720p视频设置80需要25G显存，设置50需要19G显存
# 720x480p视频设置80需要8G显存，设置50需要7G显存
# A10显卡每卡24GB显存，双卡设置下可设置更高
PROPAINTER_MAX_LOAD_NUM = 5  # 设置为1启用单帧处理模式
# 是否使用半精度(FP16)以节省显存
PROPAINTER_USE_FP16 = False
# 是否启用双GPU并行处理
PROPAINTER_USE_MULTI_GPU = False  # 禁用多GPU处理
# 是否强制使用单帧处理模式，不管批处理设置如何
PROPAINTER_FORCE_SINGLE_FRAME = False  # 新增: 强制使用单帧处理
# ×××××××××× InpaintMode.PROPAINTER算法设置 end ××××××××××

# ×××××××××× InpaintMode.LAMA算法设置 start ××××××××××
# 是否开启极速模式，开启后不保证inpaint效果，仅仅对包含文本的区域文本进行去除
LAMA_SUPER_FAST = True
# ×××××××××× InpaintMode.LAMA算法设置 end ××××××××××
# ×××××××××××××××××××× [可以改] end ××××××××××××××××××××
