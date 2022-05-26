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
import rbae
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_scheduler(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def salt_pepper_noise(X, p=0.5, p2=0.95, p3=0.2):
    idx = list(range(X.shape[1]))
    np.random.shuffle(idx)
    remove_frame_num = math.floor(X.shape[1] * p3) + 1
    idx = idx[:remove_frame_num]
    X[:, idx] = 0
   
    mask = torch.rand(X.shape, dtype=torch.float32)
    mask = (mask >= p)
    X = mask * X
     
    mask2 = torch.rand(X.shape, dtype=torch.float32)
    mask2 = (mask2 >= p2)
    X = X + mask2
    X = torch.where(X > 0, 1.0, 0.0).type(torch.float32)
    
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


def draw_fig_with_time(iter, X, label, fig_num, H, W, dir):
    X = X.cpu().detach().numpy().reshape((-1, H*W))
    label = label.cpu().numpy().reshape((-1, H*W))
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



def train_rbae(save_dir, dataset_name, H, W, T_max, cortical_delay, hidden_size, batch_T=10, batch_size=32, num_epoch=300, lr=0.1,
              log_iter=False, max_unchange_epoch=20, fig_dir='./tmp_img/bars_'):
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

    print("SUCCESSFULLY LOAD DATASET!")

    net = rbae.rbae(H * W, hidden_size).to(device)  
    
    ceriterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    min_ver_loss = Inf
    unchange_epoch = 0

    for epoch in range(num_epoch):
        running_loss = 0
        total_loss = 0

        for iter, (X, _) in enumerate(train_loader):
            X = X.reshape(X.shape[0], T_max, -1)
            label = copy.deepcopy(X)  
            X = X[:, :T_max-cortical_delay]
            label = label[:, cortical_delay:]
            X = salt_pepper_noise(X)

            X = X.to(device)
            label = label.to(device)
            
            pre_spikes = nn.Parameter(torch.zeros((X.shape[0], hidden_size[0][-1])), requires_grad=False).to(device)
            
            for t_0 in np.arange(0, T_max-cortical_delay, batch_T):       
                outputs, h = net(X[:, t_0:min(t_0+batch_T, T_max-cortical_delay)], pre_spikes)
                labels = label[:, t_0:min(t_0+batch_T, T_max-cortical_delay)]
                loss = ceriterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.cpu().item()
                total_loss += loss.cpu().item()
                
                pre_spikes = h.detach()

                if iter == 0 and epoch % 10 == 0 and epoch != 0 and t_0 == 0:
                    draw_fig_with_time(epoch, outputs, labels, batch_size, H, W, dir=fig_dir)

            if (iter % 10 == 0) and log_iter:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epoch, iter + 1, len(train_dataset) // batch_size, running_loss))
                running_loss = 0

        if epoch == 60:
            optimizer = lr_scheduler(optimizer)
        if epoch == 150:
            optimizer = lr_scheduler(optimizer)

        print("After training epoch [%d], loss [%.5f]" % (epoch, total_loss))

        with torch.no_grad():
            cur_ver_loss = 0
            for iter, (X, _) in enumerate(ver_loader):
                X = X.reshape(X.shape[0], T_max, -1)
                label = copy.deepcopy(X)  
                X = X[:, :T_max-cortical_delay]
                label = label[:, cortical_delay:]

                X = X.to(device)
                label = label.to(device)

                pre_spikes = torch.zeros((X.shape[0], hidden_size[0][-1])).to(device)
                outputs, h = net(X, pre_spikes)

                loss = ceriterion(outputs, label)
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


def train_moving_shapes():
    H = 28          
    W = 28        
    T_max = 1000    
    train_rbae(save_dir="./tmp_net/moving_shapes_rbae_net_2.pty", dataset_name="moving_shapes_large", 
                H=H, W=W, T_max=T_max, cortical_delay=1, batch_T=20, hidden_size=[[600, 300], [300, 600]], log_iter=False, fig_dir='./tmp_img/moving_shape/moving_shape_rbae_2_', lr = 0.001, num_epoch=300)

if __name__ == "__main__":
    train_moving_shapes()
