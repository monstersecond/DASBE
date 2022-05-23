import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from clrnet import CLRNET
from dataset_utils import gain_dataset
from FCA import HC
from analysis import step_wise_analysis_dynamics, draw_spikes_dynamic, draw_spikes_dynamic_2, labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping, autocorrelation, coloring, k_means_var, victor_purpura_disssimilarity, draw_spikes, draw_context,LFP, explore_timescale, DBSCAN_cluster, VP_silhouette, step_wise_analysis, feature_neuron, find_timescale_with_silhouette


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams['savefig.dpi'] = 1000 #图片像素 
plt.rcParams['figure.dpi'] = 1000 #分辨率

Vth = 1+0.00001
f_factor = 0.9
M_decay = 0.95
M_lr = 0.01 # 0.07 for shapes


def STP(M,s):
    M = M_decay*M + M_lr*torch.outer(s,s)
    return M


def clustering_dynamic(x, bestnet, labels, refractory = 9,cortical_delay=29, iterations=30):
    s_refractory = refractory
    a_refractory = refractory
    net = bestnet
    T_max = x.shape[0]      # 这张图的时间窗长度
    W = x.shape[1]          # 图片宽度
    H = x.shape[2]          # 图片高度
    x = torch.tensor(x, device=device)
    s = s_pre = s_prre = torch.zeros((W, H), device=device)
    f = torch.zeros((W,H),device = device)
    context_input = torch.abs(torch.randn((cortical_delay,W,H), device=device)) # gamma
    context_input /= context_input.sum(0)
    new_context_input = torch.zeros((cortical_delay, W, H),device = device)
    spike_record = []
    context_record = []
    encoding_record = []
    sil_record = []
    color_record = []
    group_record = []
    rate_record = []

    cur_time = 0
    hidden_layer = torch.zeros((cortical_delay, 300)).to(device)

    for iter in range(iterations):
        spike_record.append([])
        context_record.append([])
        encoding_record.append([])
        
        for t in range(cortical_delay):
            mem = x[cur_time] * context_input[t]
            noise = torch.rand(s.size(), device=device)
            
            # 处理refractory
            f -= 1
            s = torch.where(((mem+noise) > Vth) & (f < 0), 1, 0).type(dtype=torch.float32)
            f = torch.where(s>0, s_refractory*torch.ones((W, H), device=device), f).type(dtype=torch.float32)

            accumulate_input = torch.where(s+s_pre+s_prre>0, 1, 0)

            output, hidden_layer[t] = net(accumulate_input.reshape(1, 1, -1).type(dtype=torch.float32), hidden_layer[t])

            s_prre = s_pre * torch.tensor(np.random.choice([0, 1], s.shape, p=[0.5, 0.5]), device=device)
            s_pre = s * torch.tensor(np.random.choice([0, 1], s.shape, p=[0.5, 0.5]), device=device)
            new_context_input[t,:,:] = output.reshape(W, H) 
            encoding_record[-1].append(np.array(hidden_layer[t].reshape(20, 15).cpu().detach()))
            spike_record[-1].append(np.array(s.cpu()))
            context_record[-1].append(np.array(output.reshape(W, H).detach().cpu()))
            
        cur_time += 1  # 更新当前的时间步

        context_input = new_context_input
        group, syn, rate, color = step_wise_analysis_dynamics(np.array(spike_record[-1]), labels[iter], [cortical_delay, refractory])
        group_record.append(group)
        sil_record.append(syn)
        color_record.append(color)
        rate_record.append(rate)

    spike_record = np.array(spike_record)
    context_record = np.array(context_record)
    encoding_record = np.array(encoding_record)

    # np.save("./tmp_fig_data/spike_record_dynamic.npy", spike_record)
    
    # draw_spikes_dynamic(spike_record, np.array(x.cpu()), name='low_level')
    rng = []
    rng.extend(list(range(2)))
    rng.extend(list(range(12, 18)))
    rng.extend(list(range(27, 33)))
    rng.extend(list(range(42, 48)))
    draw_spikes_dynamic_2(spike_record, labels, name='low_level_input', rng=rng)
    # draw_spikes_dynamic(context_record, np.array(x.cpu()), name='low_level_cortex')
    # LFP(context_record)
    draw_step_wise_analysis_half(group_record, sil_record, rate_record, color_record, labels, x.cpu().numpy())
    return spike_record, encoding_record


def draw_spike_rasters(spike_record,img, name = ''):
    print("shape of spike record:  ", spike_record.shape)
    iter = spike_record.shape[0]
    T = spike_record.shape[1]
    spike_record_1d = spike_record.reshape(iter,T,-1)
    size = spike_record_1d.shape[2]
    print(size)
    spike_record_1d = spike_record_1d.reshape(-1, size)
    # plt.imshow(spike_record)
    # plt.savefig('./spike_raster.png')
    # plt.show()
    def get_scatter_data(spk):
        imx = []
        imt = []
        # print("len(spk)[note:win = 32]:", len(spk))
        for i in range(len(spk)):
            for j in range(len(spk[i])):
                if spk[i][j] > 0:
                    imx.append(j)
                    imt.append(i)
        return imx, imt

    plt.figure(1,figsize=(10, 10))
    imx,imt = get_scatter_data(spike_record_1d)
    plt.scatter(imt, imx, alpha=0.2, s=4)
    plt.savefig('./'+name+'spike_raster.png')
    plt.show()

    spike_record_3d = spike_record.reshape(-1, spike_record.shape[2],spike_record.shape[3])
    def get_scatter_data_3d(spk):
        imx = []
        imy = []
        imt = []
        # print("len(spk)[note:win = 32]:", len(spk))
        for i in range(spk.shape[0]):
            for j in range(len(spk[i,:, 0])):
                for k in range(len(spk[i,0, :])):
                    # print("spk[i][j, k]:",spk[i][j, k])
                    if spk[i, j, k] > 0:
                        imx.append(j)
                        imy.append(k)
                        imt.append(i)
        return np.array(imx), np.array(imy), np.array(imt)


def draw_step_wise_analysis(grp,syn,r,clr):
    n = len(grp)
    ncol = np.ceil(n/10)
    print(n)
    plt.figure()
    for i in range(n):
        plt.subplot(11,2*ncol,2*i+1)
        plt.imshow(grp[i])
        plt.axis('off')
        plt.box('off')
        plt.subplot(11,2*ncol,2*i+2)
        print(clr[i].shape)
        plt.imshow(clr[i][0,:, :, :], interpolation='nearest')
        plt.axis('off')
        plt.box('off')
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(n),np.array(syn))
    plt.ylim((0,1))
    plt.subplot(2,1,2)
    plt.plot(np.arange(n), np.array(r))
    plt.ylim((-1, 1))
    plt.show()
    # np.save("./tmp_fig_data/step_wise_analysis_n.npy", n)
    # np.save("./tmp_fig_data/step_wise_analysis_syn.npy", np.array(syn))
    # np.save("./tmp_fig_data/step_wise_analysis_r.npy", np.array(r))


def draw_step_wise_analysis_half(grp, syn, r, clr, label, input_img):
    n = len(grp)
    ncol = 20
    nrow = np.ceil(n / ncol)
    print(n)
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 将高度和宽度百分比缩小到零
    # sns.set_theme(style='darkgrid')
    fig_id = 1
    for i in range(n):
        plt.subplot(nrow * 2, ncol, (i // ncol) * ncol * 2 + i % ncol + 1)
        colors = ['white', 'gold', 'royalblue', 'limegreen'] 
        shape_num = label[i].max() + 1
        cmap_obj = mpl.colors.ListedColormap(colors[:int(shape_num)])
        plt.imshow(grp[i], cmap=cmap_obj)
        # mask = (grp[i] == 0)
        # sns.heatmap(grp[i], mask=mask, square=True, cmap='viridis_r',
        #         xticklabels=False, yticklabels=False, cbar=False)
        fig = plt.gca()
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.get_xaxis().set_visible(False)
        plt.show()
        fig_id += 1

        # sns.set_theme(style='whitegrid')
        plt.subplot(nrow * 2, ncol, (i // ncol) * ncol * 2 + i % ncol + ncol + 1)
        colors = ['white', 'black'] 
        cmap_obj = mpl.colors.ListedColormap(colors)
        plt.imshow(input_img[i], cmap=cmap_obj)
        # mask = (label[i] == 0)
        # sns.heatmap(input_img[i], mask=mask, square=True, cmap='viridis_r',
        #         xticklabels=False, yticklabels=False, cbar=False)
        fig = plt.gca()
        fig.axes.get_yaxis().set_visible(False)
        fig.axes.get_xaxis().set_visible(False)
        plt.show()
        fig_id += 1
    plt.savefig('./draw_step_wise_analysis_half.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(n),np.array(syn))
    plt.ylim((0,1))
    plt.subplot(2,1,2)
    plt.plot(np.arange(n), np.array(r))
    plt.ylim((-1, 1))
    plt.show()
    # np.save("./tmp_fig_data/step_wise_analysis_n.npy", n)
    # np.save("./tmp_fig_data/step_wise_analysis_syn.npy", np.array(syn))
    # np.save("./tmp_fig_data/step_wise_analysis_r.npy", np.array(r))


if __name__ == '__main__':
    net = torch.load('./tmp_net/moving_shapes_rbae_net_2.pty')
    
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "moving_shapes_3")

    # random choose or assign a specific ID
    i = 267  
    # i = np.random.randint(0, multi.shape[0])
    print('CURRENT TRACE ID: ', i)

    spk, enc = clustering_dynamic(np.array(multi[i], dtype=np.float32), net, multi_label[i], refractory=9, cortical_delay=32, iterations=60)
