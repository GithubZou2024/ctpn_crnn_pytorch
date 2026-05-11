import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
LOCAL_ROOT = 'E:/programming/share/python'

def get_path(path):
    """获取完整路径"""
    if IS_KAGGLE:
        # Kaggle环境：直接返回原路径
        return path
    else:
        # 本地环境：拼接到 LOCAL_ROOT，并统一路径分隔符
        # 移除 path 开头的斜杠（如果有）
        if path.startswith('/'):
            path = path[1:]
        # 拼接并转换为本地路径格式
        full_path = os.path.join(LOCAL_ROOT, path)
        return full_path