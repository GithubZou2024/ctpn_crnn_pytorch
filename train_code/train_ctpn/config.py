# -*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
# @Author: Greg Gao(laygin)
#'''
import os

# base_dir = 'path to dataset base dir'
base_dir = './images'
img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

# Kaggle数据集路径（根据实际导入的数据集名称修改）
kaggle_root = 'E:/programming/share/python'
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if IS_KAGGLE:
    icdar15_img_dir = '/kaggle/input/icdar2015/ch4_training_images'
    icdar15_gt_dir = '/kaggle/input/icdar2015/ch4_training_localization_transcription_gt'
else:
    icdar15_img_dir = kaggle_root+'/kaggle/input/datasets/hafizshehbazali/icdar2015/ch4_training_images'
    icdar15_gt_dir = kaggle_root+'/kaggle/input/datasets/hafizshehbazali/icdar2015/ch4_training_localization_transcription_gt'
# print(icdar15_gt_dir,icdar15_gt_dir)
num_workers = 2
pretrained_weights = '/kaggle/input/datasets/pretriained-ctpn'

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
checkpoints_dir = '/kaggle/working/checkpoints'
outputs = '/kaggle/working/logs'

# 自动创建必要的目录
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(outputs, exist_ok=True)