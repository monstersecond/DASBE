from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time
import matplotlib.pyplot as plt
import math

import dataset.plot_tools as plot_tools
import dataset_utils
import clrnet
import nt_xent
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
#     # if epoch % lr_decay_epoch == 0 and epoch > 1:
#     for param_group in optimizer.param_groups:
#         if param_group['lr'] * 0.1 >= 1e-3:
#             param_group['lr'] = param_group['lr'] * 0.1
#     return optimizer,dtype=torch.float32


def lr_scheduler(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def salt_pepper_noise(X, p=0.5):
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= p)
    X = mask * X
    return X


def draw_fig(iter, X, label, fig_num, H, W, dir):
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
        label = label.cpu().numpy()
    ncols = 6
    nrows = math.ceil(fig_num / 6)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows * 2, figsize=(16, 5 * nrows))

    for i in range(nrows * 2):
        for j in range(ncols):
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['bottom'].set_visible(False)
            axes[i][j].spines['left'].set_visible(False)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            # print(X[idx].reshape(H, W))
            plot_tools.plot_input_image(X[idx].reshape(H, W), axes[i][j])
            idx += 1
            if idx >= fig_num:
                break

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            plot_tools.plot_input_image(label[idx].reshape(H, W), axes[i + nrows][j])
            idx += 1
            if idx >= fig_num:
                break

    fig.savefig(dir + str(iter) + '.png')


def train_clrnet(save_dir, dataset_name, H, W, hidden_size, batch_size=128, num_epoch=150, lr=0.01,
              log_iter=False, max_unchange_epoch=25, temperature=0.5, fig_dir='./tmp_img/bars_'):
    """
    Train BAE network.

    Inputs:
        batch_size: batch size
        num_epoch: maximal epoch
        lr: initial learning rate
        log_iter: whether log info of each iteration in one epoch
        max_unchange_epoch: maximal epoch number unchanged verification loss, exceed will stop training
    Outputs:
    """
    train_dataset = dataset_utils.ClrNetDataset("./tmp_data", dataset_name, is_single=True,
                                                 train=True, ver=True, is_ver=False)
    val_dataset = dataset_utils.ClrNetDataset("./tmp_data", dataset_name, is_single=True,
                                               train=True, ver=True, is_ver=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)
    ver_loader = DataLoader(dataset=val_dataset, batch_size=1)

    net = clrnet.CLRNET(H * W, hidden_size).to(device)  # 输入大小为28*28，隐藏层大小为500
    print(net)

    ceriterion_1 = nt_xent.NT_Xent(3, temperature)
    ceriterion_2 = F.binary_cross_entropy  # 交叉熵损失函数#nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0

        for iter, (x_i, x_j) in enumerate(train_loader):
            x_i = x_i.reshape(x_i.shape[1], -1)
            x_j = x_j.reshape(x_j.shape[1], -1)

            label_i = copy.deepcopy(x_i)  # reconstruct
            label_j = copy.deepcopy(x_j)

            x_i = salt_pepper_noise(x_i, 0.6)
            x_j = salt_pepper_noise(x_j, 0.6)

            x_i = x_i.to(device)
            x_j = x_j.to(device)
            label_i = label_i.to(device)
            label_j = label_j.to(device)

            out_i, spikes_i, mem_i = net(x_i)
            out_j, spikes_j, mem_j = net(x_j)

            loss1 = ceriterion_1(spikes_i, spikes_j)
            loss2 = ceriterion_2(out_i, label_i)
            loss3 = ceriterion_2(out_j, label_j)
            loss4 = spikes_i.mean() + spikes_j.mean()

            loss = 2 * loss1 + loss2 + loss3 + 0.5 * loss4
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()

            if iter == 0 and epoch % 10 == 0:# and epoch != 0:
                draw_fig(epoch, np.concatenate([out_i.cpu().detach().numpy(), out_j.cpu().detach().numpy()], axis=0), np.concatenate([label_i.cpu().detach().numpy(), label_j.cpu().detach().numpy()], axis=0), 6, H, W, dir=fig_dir)

        # 更新学习率
        if epoch == 30:
            optimizer = lr_scheduler(optimizer)
        if epoch == 50:
            optimizer = lr_scheduler(optimizer)
        if epoch == 70:
            optimizer = lr_scheduler(optimizer)

        # 输出EPOCH训练信息
        print("After training epoch [%d], loss [%.5f], hidden loss [%.5f]" % (epoch, total_loss, loss1))

        # 验证集
        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (x_i, x_j) in enumerate(ver_loader):
                x_i = x_i.reshape(x_i.shape[1], -1)
                x_j = x_j.reshape(x_j.shape[1], -1)

                label_i = copy.deepcopy(x_i)  # reconstruct
                label_j = copy.deepcopy(x_j)

                x_i = x_i.to(device)
                x_j = x_j.to(device)
                label_i = label_i.to(device)
                label_j = label_j.to(device)

                out_i, spikes_i, mem_i = net(x_i)
                out_j, spikes_j, mem_j = net(x_j)

                loss1 = ceriterion_1(spikes_i, spikes_j)
                loss2 = ceriterion_2(out_i, label_i)
                loss3 = ceriterion_2(out_j, label_j)
                loss4 = spikes_i.mean() + spikes_j.mean()

                loss = 2 * loss1 + loss2 + loss3 + 0.5 * loss4
                # print(loss1.cpu().item(), loss2.cpu().item(), loss3.cpu().item())
                cur_ver_loss += loss.cpu().item()

            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
                torch.save(net, save_dir)
            else:
                unchange_epoch += 1

        # 输出EPOCH验证集信息
        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))

        # if unchange_epoch > max_unchange_epoch:
        #     break
        if epoch % 10 == 0:
            torch.save(net, save_dir)

    torch.save(net, save_dir)


def train_clrnet_with_batch(save_dir, dataset_name, H, W, hidden_size, batch_size=128, num_epoch=150, lr=0.01,
              log_iter=False, max_unchange_epoch=25, temperature=0.5, fig_dir='./tmp_img/bars_'):
    """
    Train BAE network.

    Inputs:
        batch_size: batch size
        num_epoch: maximal epoch
        lr: initial learning rate
        log_iter: whether log info of each iteration in one epoch
        max_unchange_epoch: maximal epoch number unchanged verification loss, exceed will stop training
    Outputs:
    """
    train_dataset_0 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                                 train=True, ver=True, is_ver=False, shape_id=0)
    train_dataset_1 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                                 train=True, ver=True, is_ver=False, shape_id=1)
    train_dataset_2 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                                 train=True, ver=True, is_ver=False, shape_id=2)
    val_dataset_0 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                               train=True, ver=True, is_ver=True, shape_id=0)
    val_dataset_1 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                               train=True, ver=True, is_ver=True, shape_id=1)
    val_dataset_2 = dataset_utils.ClrNetBatchDataset("./tmp_data", dataset_name, is_single=True,
                                               train=True, ver=True, is_ver=True, shape_id=2)

    train_loader_0 = DataLoader(dataset=train_dataset_0, batch_size=batch_size, shuffle=True, num_workers=2)
    train_loader_1 = DataLoader(dataset=train_dataset_1, batch_size=batch_size, shuffle=True, num_workers=2)
    train_loader_2 = DataLoader(dataset=train_dataset_2, batch_size=batch_size, shuffle=True, num_workers=2)
    ver_loader_0 = DataLoader(dataset=val_dataset_0, batch_size=batch_size)
    ver_loader_1 = DataLoader(dataset=val_dataset_1, batch_size=batch_size)
    ver_loader_2 = DataLoader(dataset=val_dataset_2, batch_size=batch_size)

    net = clrnet.CLRNET(H * W, hidden_size).to(device)  # 输入大小为28*28，隐藏层大小为500
    print(net)

    ceriterion_1 = nt_xent.NT_Xent_batch(temperature)
    ceriterion_2 = F.binary_cross_entropy  # 交叉熵损失函数#nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0
        contrust_loss = 0

        for iter, (x_0, x_1, x_2) in enumerate(zip(train_loader_0, train_loader_1, train_loader_2)):
            xs = []
            labels = []
            outs = []
            spikes = []
            mems = []
            xs.append(x_0)
            xs.append(x_1)
            xs.append(x_2)

            for i in range(3):
                xs[i] = xs[i].reshape(xs[i].shape[0], -1)
                labels.append(copy.deepcopy(xs[i]))
                xs[i] = salt_pepper_noise(xs[i], 0.6)
                xs[i] = xs[i].to(device)
                labels[i] = labels[i].to(device)
                out_, spikes_, mem_ = net(xs[i])
                outs.append(out_)
                spikes.append(spikes_)
                mems.append(mem_)

            loss1 = ceriterion_1(spikes[0], spikes[1], spikes[2])
            loss2 = ceriterion_2(outs[0], labels[0])
            loss3 = ceriterion_2(outs[1], labels[1])
            loss4 = ceriterion_2(outs[2], labels[2])
            # loss5 = spikes[0].mean() + spikes[1].mean() + spikes[2].mean()
            
            loss = loss1 + loss2 + loss3 + loss4 #+ 0.1 * loss5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()
            contrust_loss += loss1.cpu().item()

            if iter == 0 and epoch % 10 == 0:# and epoch != 0:
                draw_fig(epoch, np.concatenate([outs[i][:6].cpu().detach().numpy() for i in range(3)], axis=0), 
                        np.concatenate([labels[i][:6].cpu().detach().numpy() for i in range(3)], axis=0), 
                        18, H, W, dir=fig_dir)

        # 更新学习率
        # if epoch == 50:
        #     optimizer = lr_scheduler(optimizer)
        # if epoch == 100:
        #     optimizer = lr_scheduler(optimizer)
        # if epoch == 150:
        #     optimizer = lr_scheduler(optimizer)

        # 输出EPOCH训练信息
        print("After training epoch [%d], loss [%.5f], hidden loss [%.5f]" % (epoch, total_loss, contrust_loss))

        # 验证集
        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (x_0, x_1, x_2) in enumerate(zip(ver_loader_0, ver_loader_1, ver_loader_2)):
                xs = []
                labels = []
                outs = []
                spikes = []
                mems = []
                xs.append(x_0)
                xs.append(x_1)
                xs.append(x_2)

                for i in range(3):
                    xs[i] = xs[i].reshape(xs[i].shape[0], -1)
                    labels.append(copy.deepcopy(xs[i]))
                    xs[i] = salt_pepper_noise(xs[i], 0.6)
                    xs[i] = xs[i].to(device)
                    labels[i] = labels[i].to(device)
                    out_, spikes_, mem_ = net(xs[i])
                    outs.append(out_)
                    spikes.append(spikes_)
                    mems.append(mem_)

                loss1 = ceriterion_1(spikes[0], spikes[1], spikes[2])
                loss2 = ceriterion_2(outs[0], labels[0])
                loss3 = ceriterion_2(outs[1], labels[1])
                loss4 = ceriterion_2(outs[2], labels[2])
                # loss5 = spikes[0].mean() + spikes[1].mean() + spikes[2].mean()
            
                loss = loss1 + loss2 + loss3 + loss4 #+ 0.1 * loss5

                cur_ver_loss += loss.cpu().item()

            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
                torch.save(net, save_dir)
            else:
                unchange_epoch += 1

        # 输出EPOCH验证集信息
        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))

        # if unchange_epoch > max_unchange_epoch:
        #     break

        if epoch % 10 == 0:
            torch.save(net, save_dir)

    torch.save(net, save_dir)


def train_shapes():
    H = 28
    W = 28
    train_clrnet(save_dir="./tmp_net/shape_clrnet_spikes_net.pty", dataset_name="shapes", H=H, W=W, hidden_size=[[500, 1200], [1200, 500]],
               log_iter=False, batch_size=3, fig_dir='./tmp_imgs/clrnet/shapes_clrnet_spikes_net_')


def train_shapes_batch_sparse():
    H = 28
    W = 28
    train_clrnet_with_batch(save_dir="./tmp_net/shape_clrnet_spikes_batch_sparse_net.pty", dataset_name="shapes", H=H, W=W, hidden_size=[[500, 350, 1500], [1500, 350, 500]],
               log_iter=False, num_epoch=500, batch_size=512, fig_dir='./tmp_imgs/clrnet_batch/shapes_clrnet_spikes_batch_sparse_net_', lr=1e-3)


def train_shapes_batch():
    H = 28
    W = 28
    train_clrnet_with_batch(save_dir="./tmp_net/shape_clrnet_spikes_batch_net4.pty", dataset_name="shapes", H=H, W=W, hidden_size=[[600, 400, 350, 1600], [1600, 350, 400, 600]],
               log_iter=False, max_unchange_epoch=100, batch_size=512, num_epoch=500, fig_dir='./tmp_imgs/clrnet_batch/shapes_clrnet_spikes_batch_net4_', lr=1e-3)



if __name__ == "__main__":
    # train_shapes()
    train_shapes_batch()
