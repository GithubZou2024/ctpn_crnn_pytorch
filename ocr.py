import cv2
from math import *
import numpy as np
from detect.ctpn_predict import get_det_boxes
from recognize.crnn_recognizer import PytorchOcr
recognizer = PytorchOcr()

def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


def charRec(img, text_recs, adjust=False):
    """
    加载OCR模型，进行字符识别
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]

    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
        # dis(partImg)
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue
        text = recognizer.recognize(partImg)
        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)  # 识别文字

    return results

def enhance_image(image):
    """图像增强，提高 CTPN 检测率"""
    import cv2
    import numpy as np
    
    # 转为灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. 直方图均衡化（增强对比度）
    gray = cv2.equalizeHist(gray)
    
    # 2. 二值化（可选，根据情况）
    # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 降噪
    gray = cv2.medianBlur(gray, 3)
    
    # 转回 RGB
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return enhanced

def pad_to_size(image, target_height=600, target_width=None):
    """
    保持原图比例，四周填充到目标尺寸
    Args:
        image: 输入图片
        target_height: 目标高度
        target_width: 目标宽度（如果不指定，按原图宽高比计算）
    Returns:
        padded: 填充后的图片
        scale: 缩放比例（这里是1，因为没有缩放）
        pad_top, pad_left: 填充的尺寸（用于映射坐标）
    """
    h, w = image.shape[:2]
    
    # 如果不指定宽度，按原图宽高比计算目标宽度
    if target_width is None:
        target_width = int(w * (target_height / h))
    
    # 计算缩放比例（保持原图不变，只是填充）
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # 取较小的比例，确保图片完整
    
    # 缩放图片（如果比例不是1）
    if scale != 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    else:
        new_w, new_h = w, h
    
    # 计算填充
    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left
    
    # 填充（用白色或黑色）
    padded = cv2.copyMakeBorder(
        image, 
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, 
        value=(255, 255, 255)  # 白色填充
    )
    
    return padded, scale, pad_left, pad_top


def ocr(image, target_height=600):
    """OCR 主函数（使用填充而不是缩放）"""
    # 预处理
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # 图像增强
    # image = enhance_image(image)
    
    # 保持原图比例，填充到目标尺寸
    image, scale, pad_left, pad_top = pad_to_size(image, target_height=target_height)
    
    # 检测
    text_recs, img_framed, image = get_det_boxes(image)
    
    # 把检测框坐标映射回原图（减去填充偏移）
    if scale != 1.0 or pad_left != 0 or pad_top != 0:
        try:
            if text_recs and len(text_recs) > 0:
                new_text_recs = []
                for box in text_recs:
                    # 减去填充，除以缩放比例
                    new_box = [
                        int((box[0] - pad_left) / scale),
                        int((box[1] - pad_top) / scale),
                        int((box[2] - pad_left) / scale),
                        int((box[3] - pad_top) / scale),
                        int((box[4] - pad_left) / scale),
                        int((box[5] - pad_top) / scale),
                        int((box[6] - pad_left) / scale),
                        int((box[7] - pad_top) / scale),
                    ]
                    new_text_recs.append(new_box)
                text_recs = new_text_recs
        except:
            pass
    
    text_recs = sort_box(text_recs)
    result = charRec(image, text_recs)
    
    return result, img_framed