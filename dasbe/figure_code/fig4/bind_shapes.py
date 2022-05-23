import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
import sys
sys.path.append("../../")

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

Vth = 1+0.00001


def clustering(x, bestnet, labels, cortical_delay, iterations,refractory):
    net = bestnet
    W = x.shape[0]
    H = x.shape[1]
    x = torch.tensor(x, device = device)
    s = s_pre = s_prre = torch.zeros((W,H),device = device)
    f = torch.zeros((W,H),device = device)
    context_input = torch.abs(torch.randn((cortical_delay,W,H), device = device)) # gamma
    context_input /= context_input.sum(0)
    new_context_input = torch.zeros((cortical_delay, W, H),device = device)
    spike_record = []
    context_record = []
    encoding_record = []
    sil_record = []
    color_record = []
    group_record = []
    rate_record = []
    km_record = []
    inertia_record = []
    spk_always_record=[]
    syn_always_record = []
    rate_always_record = []

    for iter in range(iterations):
        spike_record.append([])
        context_record.append([])
        encoding_record.append([])
        for t in range(cortical_delay):
            mem = x*context_input[t]
            noise = torch.rand(s.size(),device = device)
            f -= 1
            s = torch.where(((mem+noise) > Vth) & (f < 0), 1, 0).type(dtype=torch.float32)
            f = torch.where(s>0,refractory*torch.ones((W,H),device=device),f).type(dtype=torch.float32)

            accumulate_input = torch.where(s+s_pre+s_prre>0,1,0)
            output,encoding,_ = net(accumulate_input.reshape(1,-1).type(dtype=torch.float32), refractory = refractory)
            s_prre = s_pre * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            s_pre = s * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            new_context_input[t,:,:] = output.reshape(W,H) #+ 0.5 * new_context_tmp
            spike_record[-1].append(np.array(s.cpu()))
            context_record[-1].append(np.array(output.reshape(W,H).detach().cpu()))

        context_input = new_context_input
        group, syn,rate,km,inertia, color = step_wise_analysis(np.array(spike_record[-1]),labels,[29,8])
        group_record.append(group)
        sil_record.append(syn)
        color_record.append(color)
        rate_record.append(rate)
        km_record.append(km)
        inertia_record.append(inertia)

    spike_record = np.array(spike_record)
    context_record=np.array(context_record)
    encoding_record = np.array(encoding_record)
    km_record = np.array(km_record)
    inertia_record = np.array(inertia_record)

    return spike_record, encoding_record, sil_record, rate_record, km_record, inertia_record, color_record


def syn_rate_plot(syn,r):
    n=len(syn)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n), np.array(syn))
    plt.ylim((0, 1))
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n), np.array(r))
    plt.ylim((-1, 1))
    plt.show()


def draw_step_wise_analysis(grp,syn,r,clr):
    n = len(grp)
    ncol = n//10
    plt.figure()
    for i in range(n):
        plt.subplot(11,2*ncol,2*i+1)
        plt.imshow(grp[i])
        plt.axis('off')
        plt.subplot(11,2*ncol,2*i+2)
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


net = torch.load('../../tmp_net/shapes_dae_0.8_400_net2.pty')

_, multi, _, _, multi_label, _ = gain_dataset("../../tmp_data", "shapes")

i = np.random.randint(0,6000)
# i = 2552
print('i=',i)
seed_list=np.random.randint(1,1000,5)
SPK = []
GROUPS = []
SELECTED = []
FMRI = []
PRED = []
CORR = []
AMI = []
SIL_TAO = []
SYN = []
RATE = []
KM = []
INER = []
COLOR = []
for seed in seed_list:
    setup_seed_pytorch(seed)
    spk, enc, syn, r, k,iner, c = clustering(np.array(multi[i], dtype=np.float32), net,multi_label[i],cortical_delay=32,iterations=30,refractory=9)
    SPK.append(spk)
    SYN.append(syn)
    RATE.append(r)
    KM.append(k)
    INER.append(iner)
    COLOR.append(c)
    groups,selected = labeled_synchrony_measure(spk,multi_label[i], num=8, show=False)
    GROUPS.append(groups)
    SELECTED.append(selected)
    fMRI = fMRI_measure(spk,multi_label[i],show=False)
    FMRI.append(fMRI)
    pred_low, label = k_means_var(spk,multi_label[i],K = [4], back=10, smooth=1, show=False)
    PRED.append(pred_low)
    auto_corr = autocorrelation(spk,multi_label[i],corr_len=20, show=False)
    CORR.append(auto_corr)
    AMI_score = evaluate_grouping(multi_label[i], pred_low)
    AMI.append(AMI_score)

results = {
    "spk":SPK, # 5 : iter X delay X W X H
    "groups": GROUPS,
    "selected":SELECTED,
    "fmri":FMRI,
    "pred":PRED,
    "corr":CORR,
    "ami":AMI,
    "sil_tao":SIL_TAO,
    "syn_score":SYN,
    "rate_score":RATE,
    "km_score" : KM,
    "inertia" : INER,
    "label": multi_label[i],
    "img":multi[i],
    "color":COLOR
}
np.save('./results',results)

