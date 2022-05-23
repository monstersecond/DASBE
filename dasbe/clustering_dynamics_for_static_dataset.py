import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
import sys

sys.path.append('dasbe')
from dataset_utils import gain_dataset
from FCA import HC
from analysis import labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping, autocorrelation, coloring, k_means_var, victor_purpura_disssimilarity, draw_spikes, draw_context,LFP, explore_timescale, DBSCAN_cluster, VP_silhouette, step_wise_analysis, feature_neuron, find_timescale_with_silhouette


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed_pytorch(3)


Vth = 1 + 0.00001
f_factor = 0.9
M_decay = 0.95
M_lr = 0.01 # 0.07 for shapes


def STP(M,s):
    M = M_decay * M + M_lr * torch.outer(s, s)
    return M


def clustering(x, bestnet, labels, cortical_delay=29, iterations=30, s_refractory=9, a_refractory=9, hidden_size=[40, 40]):
    net = bestnet
    W = x.shape[0]
    H = x.shape[1]
    x = torch.tensor(x, device = device)
    s = s_pre = s_prre = torch.zeros((W, H), device=device)
    f = torch.zeros((W, H), device=device)
    context_input = torch.abs(torch.randn((cortical_delay, W, H), device=device)) # gamma
    context_input /= context_input.sum(0)
    new_context_input = torch.zeros((cortical_delay, W, H), device=device)
    spike_record = []
    context_record = []
    encoding_record = []
    sil_record = []
    color_record = []
    group_record = []
    rate_record = []

    for _ in range(iterations):
        spike_record.append([])
        context_record.append([])
        encoding_record.append([])
        for t in range(cortical_delay):
            mem = x * context_input[t]
            noise = torch.rand(s.size(), device=device)
            f -= 1
            s = torch.where(((mem+noise) > Vth) & (f < 0), 1, 0).type(dtype=torch.float32)
            f = torch.where(s > 0, s_refractory * torch.ones((W, H), device=device), f).type(dtype=torch.float32)

            accumulate_input = torch.where(s+s_pre+s_prre > 0, 1, 0)

            output, encoding, _ = net(accumulate_input.reshape(1, -1).type(dtype=torch.float32), refractory=a_refractory)

            s_prre = s_pre * torch.tensor(np.random.choice([0, 1], s.shape, p=[0.5, 0.5]), device=device)
            s_pre = s * torch.tensor(np.random.choice([0, 1], s.shape, p=[0.5, 0.5]), device=device)
            new_context_input[t, :, :] = output.reshape(W,H) 
            
            encoding_record[-1].append(np.array(encoding.reshape(hidden_size[0], hidden_size[1]).cpu().detach()))
            spike_record[-1].append(np.array(s.cpu()))
            context_record[-1].append(np.array(output.reshape(W,H).detach().cpu()))

        context_input = new_context_input
        group, syn, rate, _, _, color = step_wise_analysis(np.array(spike_record[-1]), labels, [29, 8])
        group_record.append(group)
        sil_record.append(syn)
        color_record.append(color)
        rate_record.append(rate)

    spike_record = np.array(spike_record)
    context_record=np.array(context_record)
    encoding_record = np.array(encoding_record)

    draw_spikes(spike_record, np.array(x.cpu()), name='low_level')
    draw_context(context_record, np.array(x.cpu()))
    LFP(context_record)

    draw_step_wise_analysis(group_record, sil_record, rate_record, color_record)
    return spike_record, encoding_record


def draw_step_wise_analysis(grp,syn,r,clr):
    n = len(grp)
    ncol = n//10
    plt.figure()
    for i in range(n):
        plt.subplot(11,2*ncol,2*i+1)
        plt.imshow(grp[i])
        plt.subplot(11,2*ncol,2*i+2)
        print(clr[i].shape)
        plt.imshow(clr[i][0,:, :, :], interpolation='nearest')
        plt.axis('off')
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(n),np.array(syn))
    plt.ylim((0,1))
    plt.subplot(2,1,2)
    plt.plot(np.arange(n), np.array(r))
    plt.ylim((-1, 1))
    plt.show()


if __name__ == "__main__":
    net_name = sys.argv[1]
    dataset_name = sys.argv[2]

    net = torch.load('./dasbe/tmp_net/' + net_name) 

    _, multi, _, _, multi_label, _ = gain_dataset("./dasbe/tmp_data", dataset_name)

    # choose data index
    # i = np.random.randint(0,6000)
    i = 1109
    print('CHOOSE DATA ID: ', i)

    spk, enc = clustering(np.array(multi[i], dtype=np.float32), net, multi_label[i], s_refractory=9, a_refractory=9)

    labeled_synchrony_measure(spk, multi_label[i])
    
    fMRI_measure(spk, multi_label[i])

    pred_low, _ = k_means_var(spk, multi_label[i], K=[4], back=10, smooth=1)

    pred_high, _ = k_means_var(enc, multi_label[i], K=[4], back=10, smooth=1)
    
    decode_with_kmeanMask(net, enc, pred_high, multi[i])
    
    AMI_score = evaluate_grouping(multi_label[i], pred_low)
