import pickle as pkl
import sys
import os

# 获取当前文件所在的项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回到 ctpn_crnn_pytorch 目录

# 添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以导入了
from path_utils import get_path
# gen alphabet via label
# alphabet_set = set()
# infofiles = ['infofiles/infofile_selfcollect.txt','infofiles/infofile_train_public.txt']
# for infofile in infofiles:
#     f = open(infofile)
#     content = f.readlines()
#     f.close()
#     for line in content:
#         if len(line.strip())>0:
#             if len(line.strip().split('\t'))!=2:
#                 print(line)
#             else:
#                 fname,label = line.strip().split('\t')
#                 for ch in label:
#                     alphabet_set.add(ch)
#
# alphabet_list = sorted(list(alphabet_set))
# pkl.dump(alphabet_list,open('alphabet.pkl','wb'))
# 将原来的导入改为相对导入或动态路径

alphabet_list = pkl.load(open(get_path('/kaggle/working/ctpn_crnn_pytorch/train_code/train_crnn/alphabet.pkl'),'rb'))
alphabet = [ord(ch) for ch in alphabet_list]
alphabet_v2 = alphabet
# print(alphabet_v2)