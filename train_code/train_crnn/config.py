import keys
import os
import torch
import platform
from path_utils import get_path

# 根据环境自动调整配置
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if IS_KAGGLE:# Kaggle/Linux GPU环境
    cuda = True
    device = torch.device('cuda')
    workers = 4
    batchSize = 100
    ngpu = torch.cuda.device_count()  # 动态获取GPU数量
    print(f"运行在GPU环境，GPU数量: {ngpu}")
else:# 本地CPU/Windows环境
    cuda = False
    device = torch.device('cpu')
    workers = 0  # Windows避免多进程问题
    batchSize = 8  # 减小batch size
    ngpu = 0  # 不使用GPU
    print("运行在本地CPU环境")

# 路径配置 - get_path会自动处理路径映射
train_infofile = get_path('/kaggle/input/datasets/ravi02516/20k-synthetic-ocr-dataset/train.csv')
train_infofile_fullimg = None  # 如果没有就设为None
val_infofile = get_path('/kaggle/input/datasets/ravi02516/20k-synthetic-ocr-dataset/test.csv')
# 图片目录配置
train_img_dir = '/kaggle/input/datasets/ravi02516/20k-synthetic-ocr-dataset/files/20k train'
val_img_dir = '/kaggle/input/datasets/ravi02516/20k-synthetic-ocr-dataset/files/20k train'

# 预训练模型路径 - get_path会自动处理Kaggle和本地的路径映射
# /kaggle/input/datasets/zouhahaha/pretrained-crnn/CRNN-1010.pth
pretrained_model = get_path('/kaggle/input/datasets/zouhahaha/pretrained-crnn/CRNN-1010.pth')

# 字母表配置
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2

# 模型参数
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet) + 1
nh = 256

# 训练参数
niter = 100
lr = 0.0005
beta1 = 0.5

# 保存目录
saved_model_dir = get_path('/kaggle/working/checkpoints')
if saved_model_dir and not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir, exist_ok=True)
    print(f"创建目录: {saved_model_dir}")
# 添加 log_dir
log_dir = get_path('/kaggle/working/logs')
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    print(f"创建目录: {log_dir}")

saved_model_prefix = 'CRNN-synthetic'

# 其他配置
use_log = False
remove_blank = False
experiment = None
displayInterval = 500
n_test_disp = 10
valInterval = 500
saveInterval = 500
adam = False
adadelta = False
keep_ratio = False
random_sample = True

# 打印配置信息确认
print(f"设备: {device}")
print(f"工作进程数: {workers}")
print(f"批次大小: {batchSize}")
print(f"GPU数量: {ngpu}")
print(f"训练文件: {train_infofile}")
print(f"验证文件: {val_infofile}")
print(f"预训练模型: {pretrained_model if pretrained_model else 'None'}")