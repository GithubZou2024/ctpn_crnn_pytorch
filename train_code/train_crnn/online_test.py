import os

import torch
from torch.autograd import Variable
import utils
import mydataset
from PIL import Image
import numpy as np
import crnn as crnn
import cv2
import torch.nn.functional as F
import keys
import config

alphabet = keys.alphabet_v2
converter = utils.strLabelConverter(alphabet.copy())


def val_model(infofile,model,gpu,log_file = '20260422.log'):
    log_dir = config.log_dir
    h = open(os.path.join(log_dir, log_file), 'w')
    with open(infofile) as f:
        content = f.readlines()
        num_all = 0
        num_correct = 0
        max_print = 10

        for line in content:
            # 处理不同格式
            if '\t' in line:
                fname, label = line.split('\t')
            elif ',' in line:
                parts = line.strip().split(',', 1)
                if len(parts) != 2:
                    continue
                fname, label = parts
                # 添加图片路径
                fname = os.path.join('/kaggle/input/datasets/ravi02516/20k-synthetic-ocr-dataset/files/20k train', fname.strip())
            else:
                fname, label = line.split('g:')
                fname += 'g'
            label = label.replace('\r', '').replace('\n', '')
            img = cv2.imread(fname)
            res = val_on_image(img,model,gpu)
            res = res.strip()
            label = label.strip()
            if res == label:
                num_correct+=1
            else:
                h.write('filename:{}\npred  :{}\ntarget:{}\n'.format(fname,res, label))
                if max_print>0:
                    print('filename:{}\npred  :{}\ntarget:{}'.format(fname, res, label))
                    max_print-=1
            # else:
            #     # new_name = saved_path + fname.split('/')[-1]
            #     # shutil.copyfile(fname, new_name)
            #     wrong_results.append('res:{} / label:{}'.format(res,label))
            num_all+=1
    h.write('ocr_correct: {}/{}/{}\n'.format(num_correct,num_all,num_correct/num_all))
    print(num_correct/num_all)
    h.close()
    return num_correct, num_all

def val_on_image(img,model,gpu):
    imgH = config.imgH
    h,w = img.shape[:2]
    imgW = imgH*w//h

    transformer = mydataset.resizeNormalize((imgW, imgH), is_test=True)
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    image = Image.fromarray(np.uint8(img)).convert('L')
    image = transformer( image )
    if gpu:
        image = image.cuda()
    image = image.view( 1, *image.size() )
    image = Variable( image )

    model.eval()
    preds = model( image )

    preds = F.log_softmax(preds,2)
    conf, preds = preds.max( 2 )
    preds = preds.transpose( 1, 0 ).contiguous().view( -1 )

    preds_size = Variable( torch.IntTensor( [preds.size( 0 )] ) )
    # raw_pred = converter.decode( preds.data, preds_size.data, raw=True )
    sim_pred = converter.decode( preds.data, preds_size.data, raw=False )
    return sim_pred


if __name__ == '__main__':
    import sys
    model_path = './crnn_models/CRNN-0627-crop_48_901.pth'
    gpu = True
    if not torch.cuda.is_available():
        gpu = False

    model = crnn.CRNN(config.imgH, 1, len(alphabet) + 1, 256)
    if gpu:
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    if gpu:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if len(sys.argv)>1 and 'train' in sys.argv[1]:
        infofile = 'data_set/infofile_updated_0627_train.txt'
        print(val_model(infofile, model, gpu, '0627_train.log'))
    elif len(sys.argv)>1 and 'gen' in sys.argv[1]:
        infofile = 'data_set/infofile_0627_gen_test.txt'
        print(val_model(infofile, model, gpu, '0627_gen.log'))
    else:
        infofile = 'data_set/infofile_updated_0627_test.txt'
        print(val_model(infofile, model, gpu, '0627_test.log'))




