import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from dataset_utils import gain_dataset
from FCA import HC
from analysis import labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping
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

def HNN_clustering(x, bestnet, cortical_delay=32, iterations=50,s_refractory=5, a_refractory=5):
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
            output,encoding,_ = net(accumulate_input.reshape(1,-1).type(dtype=torch.float32), refractory = a_refractory)
            s_prre = s_pre * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            s_pre = s * torch.tensor(np.random.choice([0,1],s.shape,p=[0.5,0.5]),device=device)
            new_context_input[t,:,:] = output.reshape(W,H)
            spike_record[-1].append(np.array(s.cpu()))

        context_input = new_context_input
    spike_record = np.array(spike_record)

    return spike_record

def draw_AMI_selected_images(multi,AMI_score,idx_dict, para):
    selected_ami = [AMI_score[1], AMI_score[1000], AMI_score[2000], AMI_score[3000], AMI_score[4000], AMI_score[5000],
                    AMI_score[5999]]
    fig = plt.figure()
    for i in range(len(selected_ami)):
        plt.subplot(4, 2, i+1)
        plt.imshow(np.array(multi[idx_dict[selected_ami[i]]], dtype=np.float32))
    fig.savefig('./result shape/img_' + str(para) + '.png')
    # plt.show()

def clustering_on_each_datset(net_name,data_name,para=[32,9],back=10,smooth=2):
    net = torch.load('../../tmp_net/'+net_name)
    _, multi, _, _, multi_label, _ = gain_dataset("../../tmp_data", data_name)
    print(multi[0].shape, multi_label[0])
    AMI_mean = []
    AMI_score_5 = []
    SYN_score_5 = []
    RATE_score_5 = []
    pair5 = []
    # seed_list = np.random.randint(1,1000,5)
    seed_list = [111, 222, 333, 444, 555]
    for seed in seed_list: # different random seed
        setup_seed_pytorch(seed)
        AMI_score = []
        # syn_score = []
        # rate_score = []
        AMI_syn_rate_pair = []
        idx_dict = {}
        for i in range(6000):
            spk = HNN_clustering(np.array(multi[i], dtype=np.float32), net,cortical_delay=para[0], s_refractory=para[1],a_refractory=para[1])
            pred_low = k_means(spk,multi_label[i],show=False, back=10, smooth=smooth)
            ami_score = evaluate_grouping(multi_label[i], pred_low)
            print(data_name, " seed:", seed, ' id:', i, " AMI score:", ami_score)
            # syn,rate = syn_vs_rate_score(spk, multi_label[i])
            idx_dict[ami_score] = i
            AMI_score.append(ami_score)
            # syn_score.append(syn)
            # rate_score.append(rate)
            # AMI_syn_rate_pair.append([ami_score,syn,rate])

        AMI_score.sort()
        AMI_mean.append(np.array(AMI_score).mean())
        AMI_score_5.append(AMI_score)
        # SYN_score_5.append(syn_score)
        # RATE_score_5.append(rate_score)
        pair5.append(AMI_syn_rate_pair)
    return np.array(AMI_score_5), np.array(AMI_mean), np.array(pair5)

net_name_list=['bars_bae_net.pty','corners_bae_net.pty','shapes_dae_0.8_400_net2.pty','multi_mnist_bcdae_net_0.9.pty','mnist_shape_bae_net_bast_net_para2.pty']
data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = [[54,6],[28,8],[32,9],[29,12],[29,9]] # note: corner need 50 iter

start_time = datetime.datetime.now()
for i in range(5):
    print("i = ", i)
    AMI_score, AMI_mean, ami_syn_rate_compare = clustering_on_each_datset(net_name_list[i],data_name_list[i],para_list[i])
    np.save('./AMI_npys/ami_HNN_score_5array_'+data_name_list[i],AMI_score)
    np.save('./AMI_npys/ami_HNN_score_5mean_' + data_name_list[i], AMI_mean)
    print(data_name_list[i]+' :  ',AMI_score.shape)
end_time = datetime.datetime.now()
print("time used:", end_time-start_time)




