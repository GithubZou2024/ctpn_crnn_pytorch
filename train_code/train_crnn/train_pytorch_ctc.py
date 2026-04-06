from __future__ import print_function
import argparse
import random
import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import utils
import mydataset
import crnn as crnn
import config
from online_test import val_model
import config
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
log_filename = os.path.join('log/','loss_acc-'+config.saved_model_prefix + '.log')
if not os.path.exists('debug_files'):
    os.mkdir('debug_files')
if not os.path.exists(config.saved_model_dir):
    os.mkdir(config.saved_model_dir)
if config.use_log and not os.path.exists('log'):
    os.mkdir('log')
if config.use_log and os.path.exists(log_filename):
    os.remove(log_filename)
if config.experiment is None:
    config.experiment = 'expr'
if not os.path.exists(config.experiment):
    os.mkdir(config.experiment)

print(f"训练集 infofile: {config.train_infofile}")
print(f"验证集 infofile: {config.val_infofile}")

config.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
np.random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

# cudnn.benchmark = True
train_dataset = mydataset.MyDataset(info_filename=config.train_infofile)
assert train_dataset
if not config.random_sample:
    sampler = mydataset.randomSequentialSampler(train_dataset, config.batchSize)
else:
    sampler = None
####################
import os

# 1. 打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 2. 打印 infofile 第一行和文件是否存在
with open(config.train_infofile, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    print(f"infofile 第一行: {first_line}")
    
    # 解析出图片路径（只取第一部分，不含标签）
    if '\t' in first_line:
        img_path = first_line.split('\t')[0].strip()
    else:
        img_path = first_line.strip()
    
    # 直接用 get_path 转换路径
    full_path = config.get_path(img_path)
    print(f"完整路径: {full_path}")
    print(f"文件是否存在: {os.path.exists(full_path)}")

# 3. 列出图片目录的前几个文件（修正：应该用目录路径，不是文件路径）
img_dir_path = config.get_path("/kaggle/input/datasets/zouhahaha/recognition/ch4_training_word_images_gt")  # 或其他正确的图片目录
print(f"\n图片目录前5个文件:")
if os.path.exists(img_dir_path) and os.path.isdir(img_dir_path):
    for i, f in enumerate(os.listdir(img_dir_path)):
        if i >= 5:
            break
        print(f"  {f}")
else:
    print(f"  目录不存在或不是目录: {img_dir_path}")
####################
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(config.workers),
    collate_fn=mydataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio))

test_dataset = mydataset.MyDataset(
    info_filename=config.val_infofile, transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True))

converter = utils.strLabelConverter(config.alphabet)
criterion = CTCLoss(reduction='sum',zero_infinity=True)
# criterion = CTCLoss()
best_acc = 0.9


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
print(f"DEBUG: type = {type(config.pretrained_model)},{config.pretrained_model}")
if config.pretrained_model!='' and os.path.exists(config.pretrained_model):
    print('loading pretrained model from %s' % config.pretrained_model)
    crnn.load_state_dict(torch.load(config.pretrained_model))
else:
    crnn.apply(weights_init)

print(crnn)

# image = torch.FloatTensor(config.batchSize, 3, config.imgH, config.imgH)
# text = torch.IntTensor(config.batchSize * 5)
# length = torch.IntTensor(config.batchSize)
if config.cuda:
    # 先包装 DataParallel，再移动到 device
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        crnn = torch.nn.DataParallel(crnn)
    crnn = crnn.to(config.device)
    criterion = criterion.to(config.device)
    
# image = Variable(image)
# text = Variable(text)
# length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    num_correct, num_all = val_model(config.val_infofile, net, True, log_file='compare-' + config.saved_model_prefix + '.log')
    accuracy = num_correct / num_all

    print('ocr_acc: %f' % (accuracy))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('ocr_acc:{}\n'.format(accuracy))
    
    global best_acc, epoch
    if accuracy > best_acc:
        best_acc = accuracy
        # 修改这一行：判断是否有 module 前缀
        if hasattr(net, 'module'):
            torch.save(net.module.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, epoch, int(best_acc * 1000)))
        else:
            torch.save(net.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, epoch, int(best_acc * 1000)))
    
    # 修改这一行：保存最新模型
    if hasattr(net, 'module'):
        torch.save(net.module.state_dict(), '{}/{}.pth'.format(config.saved_model_dir, config.saved_model_prefix))
    else:
        torch.save(net.state_dict(), '{}/{}.pth'.format(config.saved_model_dir, config.saved_model_prefix))


def trainBatch(net, criterion, optimizer, train_iter):
    data = next(train_iter)
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = cpu_images.to(config.device)

    text, length = converter.encode(cpu_texts)
    # utils.loadData(text, t)
    # utils.loadData(length, l)

    preds = net(image)  # seqLength x batchSize x alphabet_size
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))  # seqLength x batchSize
    cost = criterion(preds.log_softmax(2).cpu(), text, preds_size, length) / batch_size
    if torch.isnan(cost):
        print(batch_size,cpu_texts)
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    return cost


for epoch in range(config.niter):
    loss_avg.reset()
    print('epoch {}....'.format(epoch))
    train_iter = iter(train_loader)
    i = 0
    n_batch = len(train_loader)
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost = trainBatch(crnn, criterion, optimizer, train_iter)
        print('epoch: {} iter: {}/{} Train loss: {:.3f}'.format(epoch, i, n_batch, cost.item()))
        loss_avg.add(cost)
        loss_avg.add(cost)
        i += 1
    print('Train loss: %f' % (loss_avg.val()))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
            f.write('train_loss:{}\n'.format(loss_avg.val()))

    val(crnn, test_dataset, criterion)