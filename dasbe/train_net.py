from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time
import matplotlib.pyplot as plt
import math
import random
import sys

import dataset.plot_tools as plot_tools
import bae
import dataset_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_scheduler(optimizer, beta=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * beta
    return optimizer


def salt_pepper_noise(X, p):
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= p) # 0.5
    X = mask * X
    return X


def salt_pepper_noise_range(X, p_min, p_max):
    p = random.uniform(p_min, p_max)
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= p) 
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


def train_net(save_dir, dataset_name, H, W, hidden_size, p_min=0.6, p_max=0.8, batch_size=16, num_epoch=300, lr=0.1,
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

    net = bae.bae(H * W, hidden_size).to(device)  
    print(net)

    ceriterion = F.binary_cross_entropy 
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0

        for iter, (X, _) in enumerate(train_loader):
            X = X.reshape(X.shape[0], -1)
            label = copy.deepcopy(X)  # reconstruct target
            # X = salt_pepper_noise(X, p=0.6)
            X = salt_pepper_noise_range(X, p_min, p_max)

            X = X.to(device)
            label = label.to(device)

            output, _, mem = net(X)

            loss = ceriterion(output, label)

            running_loss += loss.cpu().item()
            total_loss += loss.cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter % 10 == 0) and log_iter:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epoch, iter + 1, len(train_dataset) // batch_size, running_loss))
                running_loss = 0

            if iter == 0 and epoch % 10 == 0 and epoch != 0:
                draw_fig(epoch, output[-16:], X[-16:], 16, H, W, dir=fig_dir)

        if epoch == 60:
            optimizer = lr_scheduler(optimizer, beta=0.5)
        if epoch == 100:
            optimizer = lr_scheduler(optimizer, beta=0.5)
        if epoch == 130:
            optimizer = lr_scheduler(optimizer, beta=0.5)
        if epoch == 150:
            optimizer = lr_scheduler(optimizer, beta=0.5)
        if epoch == 170:
            optimizer = lr_scheduler(optimizer, beta=0.1)

        print("After training epoch [%d], loss [%.5f]" % (epoch, total_loss))

        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (X, _) in enumerate(ver_loader):
                X = X.reshape(X.shape[0], -1)
                label = copy.deepcopy(X)

                X = X.to(device)
                label = label.to(device)

                output, _, mem = net(X)
                loss = ceriterion(output, label)
                cur_ver_loss += loss.cpu().item()

            if cur_ver_loss < min_ver_loss:
                min_ver_loss = cur_ver_loss
                unchange_epoch = 0
                torch.save(net, save_dir)
            else:
                unchange_epoch += 1

        print("After verification epoch [%d], loss [%.5f, %.5f]" % (epoch, cur_ver_loss, min_ver_loss))

        if unchange_epoch > max_unchange_epoch:
            break

    torch.save(net, save_dir)


def train_bars():
    train_net(save_dir="./tmp_net/bars_net.pty", dataset_name="bars", H=20, W=20, hidden_size = [[100], [100]], log_iter=False, fig_dir='./tmp_imgs/bar/bars_', lr=0.01)


def train_corner():
    train_net(save_dir="./tmp_net/corners_net.pty", dataset_name="corners", H=28, W=28,hidden_size = [[100], [100]], log_iter=False, fig_dir='./tmp_imgs/corner/corners_', lr=0.01)


def train_shape():
    train_net(save_dir="./tmp_net/shapes_net.pty", batch_size=1024, dataset_name="shapes", p_min=0.6, p_max=0.8, H=28, W=28, hidden_size=[[512, 400], [400, 512]], log_iter=False, fig_dir='./tmp_imgs/shape/shapes_', lr=0.01)  


def train_mnist_shape():
    H = 28
    W = 28
    train_net(save_dir="./tmp_net/mnist_shape_net.pty", dataset_name="mnist_shape", H=H, W=W, hidden_size=[[250], [250]], log_iter=False, fig_dir='./tmp_imgs/mnist_shape/mnist_shape_', batch_size=1024, lr=0.031685, p_min=0.6, p_max=0.6)


if __name__ == "__main__":
    dataset_chosen = sys.argv[1]
    print("CHOSEN DATASET IS", dataset_chosen)

    if dataset_chosen == 'shape':
        train_shape()
    elif dataset_chosen == 'mnist_shape':
        train_mnist_shape()
    elif dataset_chosen == 'bars':
        train_bars()
    elif dataset_chosen == 'corner':
        train_corner()
