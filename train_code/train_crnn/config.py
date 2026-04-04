import keys
import os
import torch

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
KAGGLE_ROOT = r"E:\programming\share\python"

def get_path(path):
    if IS_KAGGLE:
        return path
    else:
        # 确保路径分隔符正确
        full_path = os.path.join(KAGGLE_ROOT, path)
        return full_path.replace('/', os.sep).replace('\\', os.sep)

cuda = True
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
train_infofile = get_path('/kaggle/input/datasets/zouhahaha/recognition/ch4_training_word_images_gt/ch4_training_word_images_gt.txt')
train_infofile_fullimg = get_path('')
val_infofile = get_path('/kaggle/input/datasets/zouhahaha/recognition/ch4_test_word_images_gt/ch4_test_word_images_gt.txt')
alphabet = keys.alphabet
alphabet_v2 = keys.alphabet_v2
workers = 4
batchSize = 100
imgH = 32
imgW = 280
nc = 1
nclass = len(alphabet)+1
nh = 256
niter = 100
lr = 0.0006
beta1 = 0.5
ngpu = 2
pretrained_model = get_path('/kaggle/input/pretrained_CRNN/CRNN.pth')
saved_model_dir = get_path('crnn_models')
if saved_model_dir and not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir, exist_ok=True)
    print(f"创建目录: {saved_model_dir}")
saved_model_prefix = 'CRNN-'
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