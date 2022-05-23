import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
sys.path.append('../../')

from dataset_utils import gain_dataset
from FCA import HC
from analysis import labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping, syn_vs_rate_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
import datetime

Vth = 1+0.00001
f_factor = 0.9

def setup_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def HNN_clustering(x, bestnet, label, cortical_delay=32, iterations=30,s_refractory=5, a_refractory=5, step_wise=False):
    net = bestnet
    #print(net)
    W = x.shape[0]
    H = x.shape[1]
    x = torch.tensor(x, device = device)
    s = s_pre =s_prre = torch.zeros((W,H),device = device) #spree=
    f = torch.zeros((W,H),device = device)
    context_input = torch.abs(torch.randn((cortical_delay,W,H), device = device)) # gamma
    context_input /= context_input.sum(0) #
    new_context_input = torch.abs(torch.randn((cortical_delay, W, H),device = device))
    spike_record = []
    encoding_record = []
    syn_record = []
    rate_recore = []
    km_record = []
    inertia_record=[]

    for iter in range(iterations):
        spike_record.append([])
        encoding_record.append([])
        for t in range(cortical_delay):
            mem = x*context_input[t]
            noise = torch.rand(s.size(),device = device)
            f -= 1
            s = torch.where(((mem + noise) > Vth) & (f < 0), 1, 0).type(dtype=torch.float32)
            f = torch.where(s>0,s_refractory*torch.ones((W,H),device=device),f).type(dtype=torch.float32)
            accumulate_input = torch.where(s+s_pre+s_prre>0,1,0)
            output, _, _ = net(accumulate_input.reshape(1, -1).type(dtype=torch.float32), refractory = a_refractory)
            s_prre = s_pre * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            s_pre = s * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            new_context_input[t,:,:] = output.reshape(W,H)
            spike_record[-1].append(np.array(s.cpu()))

        context_input = new_context_input

        if step_wise:
            syn, rate, km,inertia = syn_vs_rate_score(np.array(spike_record[-1]), label, step_wise=True)
            syn_record.append(syn)
            rate_recore.append(rate)
            km_record.append(km)
            inertia_record.append(inertia)

    spike_record = np.array(spike_record)
    syn_record = np.array(syn_record)
    rate_recore = np.array(rate_recore)
    km_record = np.array(km_record)
    inertia_record = np.array(inertia_record)

    return spike_record, syn_record, rate_recore, km_record, inertia_record


def draw_AMI_selected_images(multi,AMI_score,idx_dict, para):
    selected_ami = [AMI_score[1], AMI_score[1000], AMI_score[2000], AMI_score[3000], AMI_score[4000], AMI_score[5000], AMI_score[5999]]
    fig = plt.figure()
    for i in range(len(selected_ami)):
        plt.subplot(4, 2, i+1)
        plt.imshow(np.array(multi[idx_dict[selected_ami[i]]], dtype=np.float32))
    fig.savefig('./result shape/img_' + str(para) + '.png')


def clustering_on_each_datset(net_name,data_name,para=[32,9],back=2,smooth=2, step_wise=False, N=100):
    net = torch.load('../../tmp_net/'+net_name)
    _, multi, _, _, multi_label, _ = gain_dataset("../../tmp_data", data_name)
    # print(multi[0].shape, multi_label[0])

    pair5 = []

    seed = 111
    setup_seed_pytorch(seed)

    syn_score = []
    rate_score = []
    km_score = []
    Inertia=[]
    AMI_syn_rate_pair = []

    random_selected_idx = np.random.randint(0,6000,N)
    t = 0
    for i in random_selected_idx:
        if t == 74:
            t = t + 1
            continue
        start_time = datetime.datetime.now()
        spk, syns, rates, kms, inertias = HNN_clustering(np.array(multi[i], dtype=np.float32), net, multi_label[i],cortical_delay=para[0], s_refractory=para[1],a_refractory=para[1], step_wise=step_wise)

        pred_low = k_means(spk,multi_label[i],show=False, back=back, smooth=smooth)
        ami_score = evaluate_grouping(multi_label[i], pred_low) # based on 2 iters

        SYN,RATE,KM,INER = syn_vs_rate_score(spk, multi_label[i], back=back) 
        end_time = datetime.datetime.now()
        print(data_name, t," time used:", end_time - start_time,' syn:',SYN,' rate:',RATE,'km',KM, 'ami', ami_score)

        syn_score.append(syns)
        rate_score.append(rates)
        km_score.append(kms)
        Inertia.append(inertias)
        AMI_syn_rate_pair.append([ami_score,SYN,RATE,KM,INER])
        t=t+1

    pair5.append(AMI_syn_rate_pair)
    return syn_score, rate_score, km_score, Inertia, np.array(pair5)

net_name_list=['bars_bae_net.pty','corners_bae_net.pty','shapes_dae_0.8_400_net2.pty','multi_mnist_bcdae_net_0.9.pty','mnist_shape_cnn3.pty']
data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = [[54,6],[28,8],[32,9],[29,12],[64,12]] #29,9

for i in range(5):
    print(para_list[i])
    syn_score, rate_score, km_score, Inertia, ami_syn_rate_compare = clustering_on_each_datset(net_name_list[i],data_name_list[i],para_list[i], back=1,step_wise=True, N = 100)

    np.save('./AMI_npys/syn_rate_AMI_pairs'+data_name_list[i],ami_syn_rate_compare) 
    np.save('./AMI_npys/syn_score_all_' + data_name_list[i], syn_score)
    np.save('./AMI_npys/rate_score_all_' + data_name_list[i], rate_score)
    np.save('./AMI_npys/km_score_all_' + data_name_list[i], km_score)
    np.save('./AMI_npys/Inertia_all_' + data_name_list[i], Inertia)
