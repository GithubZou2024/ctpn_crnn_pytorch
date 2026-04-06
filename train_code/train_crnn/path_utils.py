import os

IS_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ
LOCAL_ROOT = 'E:\\programming\\share\\python'

def get_path(path):
    if IS_KAGGLE:
        return path
    else:
        return os.path.join(LOCAL_ROOT, path.lstrip('/').replace('/','\\'))