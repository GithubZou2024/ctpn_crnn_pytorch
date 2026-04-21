# -*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
# @Author: Greg Gao(laygin)
#'''
import os
import torch

# base_dir = 'path to dataset base dir'
base_dir = './images'
img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

# Kaggle数据集路径（根据实际导入的数据集名称修改）
kaggle_root = 'E:/programming/share/python'
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if IS_KAGGLE:
    icdar15_img_dir = '/kaggle/working/ctpn_images'
    icdar15_gt_dir = '/kaggle/working/ctpn_gt'
else:
    icdar15_img_dir = kaggle_root+'/kaggle/working/ctpn_images'
    icdar15_gt_dir = kaggle_root+'/kaggle/working/ctpn_gt'
print(icdar15_img_dir,icdar15_gt_dir)
num_workers = 4

# ========== 【GPU优化部分】==========
# 设备自动检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# 根据设备调整参数
if torch.cuda.is_available():
    # GPU环境：可以增大batch size和worker数量
    batch_size = 16  # 根据你的GPU显存调整，原来可能是16
    num_workers = 4  # 增加数据加载线程
    pin_memory = True  # 加速CPU到GPU传输
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    # CPU环境：保持原有配置
    batch_size = 16  # 原来的batch size
    num_workers = 2
    pin_memory = False
    print("未检测到GPU, 使用CPU进行训练")

# ========== 【模型训练参数】==========
if IS_KAGGLE:
    pretrained_weights = '/kaggle/input/datasets/zouhahaha/pretrained-ctpn/CTPN.pth'
else:
    pretrained_weights = kaggle_root + '/kaggle/input/datasets/pretriained-ctpn/CTPN.pth'
anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

# 修改为Kaggle的working目录，并确保路径使用正斜杠
# 路径配置
if IS_KAGGLE:
    checkpoints_dir = '/kaggle/working/checkpoints'
    outputs = '/kaggle/working/logs'
else:
    checkpoints_dir = kaggle_root + '/kaggle/working/checkpoints'
    outputs = kaggle_root + '/kaggle/working/logs'

# 自动创建必要的目录
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(outputs, exist_ok=True)