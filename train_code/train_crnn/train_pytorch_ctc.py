from __future__ import print_function
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch.nn import CTCLoss
import utils
import mydataset
import crnn as crnn
import config
from online_test import val_model
import os
import datetime
import csv
from datetime import datetime

# 注释掉这行，让程序自动选择GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ========== 目录初始化（使用 config 中的路径）==========
# 日志文件路径（使用 config.log_dir）
log_filename = os.path.join(config.log_dir, 'loss_acc-' + config.saved_model_prefix + '.log')

# debug 目录
debug_dir = '/kaggle/working/ctpn_crnn_pytorch/train_code/train_crnn/debug_files'
os.makedirs(debug_dir, exist_ok=True)

# 确保目录存在（exist_ok=True 避免重复创建报错）
os.makedirs(config.saved_model_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

# 清理旧日志
if config.use_log and os.path.exists(log_filename):
    os.remove(log_filename)

# experiment 目录（如果不需要可以删除）
if config.experiment:
    os.makedirs(config.experiment, exist_ok=True)

config.manualSeed = random.randint(1, 10000)
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
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(config.workers),
    collate_fn=mydataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio))

test_dataset = mydataset.MyDataset(
    info_filename=config.val_infofile, transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True))

converter = utils.strLabelConverter(config.alphabet)
criterion = CTCLoss(reduction='sum', zero_infinity=True)
best_acc = 0.9

# ========== 添加 loss 记录文件 ==========
loss_log_path = os.path.join(config.log_dir, 'training_loss.csv')
with open(loss_log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'batch', 'batch_idx', 'total_batches', 'loss', 'timestamp', 'type'])
print(f"Loss 记录将保存到: {loss_log_path}")

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
if config.pretrained_model and os.path.exists(config.pretrained_model):
    print('loading pretrained model from %s' % config.pretrained_model)
    state_dict = torch.load(config.pretrained_model, map_location='cpu')
    crnn.load_state_dict(state_dict, strict=False)
else:
    print('no pretrained model, training from scratch')
    crnn.apply(weights_init)

print(crnn)

# ========== 多GPU支持 ==========
device = config.device
if config.cuda:
    crnn = crnn.to(device)
    if config.ngpu > 1:
        print(f"启用 DataParallel，使用 {config.ngpu} 个GPU")
        crnn = torch.nn.DataParallel(crnn)
    criterion = criterion.to(device)
    print(f"模型已加载到 {device}")
# 在创建模型后添加
print(f"Model type: {type(crnn)}")
print(f"Is DataParallel: {isinstance(crnn, torch.nn.DataParallel)}")

# loss averager
loss_avg = utils.averager()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)


def val(net, dataset, criterion, max_iter=100, current_epoch=0):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False

    # 如果是 DataParallel 包装的，取 .module
    if isinstance(net, torch.nn.DataParallel):
        net_to_val = net.module
    else:
        net_to_val = net
    
    num_correct, num_all = val_model(config.val_infofile, net_to_val, True, log_file='compare-'+config.saved_model_prefix+'.log')
    accuracy = num_correct / num_all if num_all > 0 else 0

    print('ocr_acc: %f' % (accuracy))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('ocr_acc:{}\n'.format(accuracy))
    
    global best_acc
    if accuracy > best_acc:
        best_acc = accuracy
        model_to_save = net.module if isinstance(net, torch.nn.DataParallel) else net
        torch.save(model_to_save.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, current_epoch, int(best_acc * 1000)))
    
    # 定期保存模型
    model_to_save = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save(model_to_save.state_dict(), '{}/{}.pth'.format(config.saved_model_dir, config.saved_model_prefix))
    
    # 返回准确率
    return accuracy


def trainBatch(net, criterion, optimizer, train_iter, converter, device):
    data = next(train_iter)
    cpu_images, cpu_texts = data
    image = cpu_images.to(device)

    text, length = converter.encode(cpu_texts)
    
    preds = net(image)
    
    # 获取实际 batch size
    actual_batch_size = preds.size(1)
    
    print(f"Debug: cpu_images batch size: {cpu_images.size(0)}")
    print(f"Debug: preds batch size: {actual_batch_size}")
    print(f"Debug: text length: {len(text)}")
    print(f"Debug: length length: {len(length)}")
    
    # 确保 text 和 length 与 preds 的 batch size 匹配
    if len(text) > actual_batch_size:
        text = text[:actual_batch_size]
        length = length[:actual_batch_size]
    elif len(text) < actual_batch_size:
        # 这种情况不应该发生，但为了安全
        print(f"Warning: text length {len(text)} < actual_batch_size {actual_batch_size}")
        return None
    
    text = text.to(device)
    length = length.to(device)
    
    seq_len = preds.size(0)
    preds_size = torch.full((actual_batch_size,), seq_len, dtype=torch.int32, device=device)
    
    log_probs = preds.log_softmax(2)
    
    # 添加额外的安全检查
    if torch.isnan(log_probs).any():
        print("NaN in log_probs")
        return None
        
    cost = criterion(log_probs, text, preds_size, length) / actual_batch_size
    
    if torch.isnan(cost):
        print(f"NaN detected!")
        return None
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    return cost

# ========== 训练循环 ==========
for epoch in range(config.niter):
    loss_avg.reset()
    print('epoch {}....'.format(epoch))
    train_iter = iter(train_loader)
    i = 0
    n_batch = len(train_loader)
    
    # 记录 epoch 开始时间
    epoch_start_time = datetime.now()
    
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
        cost = trainBatch(crnn, criterion, optimizer, train_iter, converter, device)
        if cost is not None:
            loss_value = cost.item()
            print('epoch: {} iter: {}/{} Train loss: {:.6f}'.format(epoch, i, n_batch, loss_value))
            loss_avg.add(cost)
            
            # 记录每个 batch 的 loss
            with open(loss_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, i, i, n_batch,
                    f"{loss_value:.6f}",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'batch'
                ])
        i += 1
    
    epoch_end_time = datetime.now()
    epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
    
    avg_loss = loss_avg.val()
    print('Epoch {} finished, Train loss: {:.6f}, Duration: {:.2f}s'.format(epoch, avg_loss, epoch_duration))
    
    # 记录每个 epoch 的平均 loss
    with open(loss_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, -1, -1, n_batch,
            f"{avg_loss:.6f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch'
        ])
    
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('{}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
            f.write('train_loss:{}\n'.format(avg_loss))

    # 验证并记录验证准确率
    val_acc = val(crnn, test_dataset, criterion, current_epoch=epoch)
    
    # 记录验证结果
    with open(loss_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, -1, -1, n_batch,
            f"{val_acc:.6f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'val_acc'
        ])

print(f"\n训练完成！Loss 记录已保存到: {loss_log_path}")