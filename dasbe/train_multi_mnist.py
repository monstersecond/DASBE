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
import bcdae
import dataset_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_scheduler(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def salt_pepper_noise(X, p=0.5):
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= 0.6)
    X = mask * X
    return X


def draw_fig(iter, X, label, fig_num, H, W, dir):
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


def train_bcdae(save_dir, dataset_name, H, W, hidden_size, batch_size=128, num_epoch=200, lr=0.0001,
              log_iter=False, max_unchange_epoch=40, fig_dir='./tmp_img/bars_'):
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
    train_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, is_single=True,
                                                 train=True, ver=True, is_ver=False)
    val_dataset = dataset_utils.BindingDataset("./tmp_data", dataset_name, is_single=True,
                                               train=True, ver=True, is_ver=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    ver_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    net = bcdae.bcdae(H * W, hidden_size).to(device)  # 输入大小为28*28，隐藏层大小为500
    print(net)
    ceriterion = F.binary_cross_entropy  # 交叉熵损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0

        for iter, (X, _) in enumerate(train_loader):
            X = X.reshape(X.shape[0], -1)
            label = copy.deepcopy(X)  # reconstruct
            X = salt_pepper_noise(X)

            X = X.to(device)
            label = label.to(device)

            output,_, mem = net(X)

            W = [net.state_dict()['encoder.eLinear{}.weight'.format(i)] for i in range(3)]
            loss = bcdae.loss_function(W, label, output,
                                     mem, lam=1e-4)

            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter % 10 == 0) and log_iter:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epoch, iter + 1, len(train_dataset) // batch_size, running_loss))
                running_loss = 0

        if epoch % 10 == 0:
            net.samples_write(label, epoch)

        if epoch == 50:
            optimizer = lr_scheduler(optimizer)
        if epoch == 80:
            optimizer = lr_scheduler(optimizer)

        print("After training epoch [%d], loss [%.5f]" % (epoch, total_loss))

        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (X, _) in enumerate(ver_loader):
                X = X.reshape(X.shape[0], -1)
                label = copy.deepcopy(X)

                X = X.to(device)
                label = label.to(device)

                output,_,mem = net(X)
                loss1 = ceriterion(output, label)
                loss2 = torch.abs(mem).mean()
                loss = loss1 + 0.001 * loss2
                cur_ver_loss += loss.cpu().item()

            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
            else:
                unchange_epoch += 1

        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))

        if unchange_epoch > max_unchange_epoch:
            break

    torch.save(net, save_dir)


def train_multi_mnist():
    H = 48 
    W = 48  
    train_bcdae(save_dir="./tmp_net/multi_mnist_net.pty", dataset_name="multi_mnist", H=H, W=W, hidden_size=500, log_iter=False, fig_dir='./tmp_imgs/multi_mnist/multi_mnist_')


if __name__ == "__main__":
    train_multi_mnist()
