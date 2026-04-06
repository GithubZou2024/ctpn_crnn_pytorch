# path_utils.py
import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
LOCAL_ROOT = "E:\\programming\\share\\python"

def get_path(path):
    if IS_KAGGLE:
        return path
    else:
        path = path.lstrip('/').replace('/', os.sep)
        full_path = os.path.join(LOCAL_ROOT, path)
        return full_path