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

def DAE_clustering(x, bestnet, cortical_delay=32, iterations=50):
    net = bestnet
    #print(net)
    W = x.shape[0]
    H = x.shape[1]
    x = torch.tensor(x, device = device)
    # s = s_pre =s_prre = torch.zeros((W,H),device = device) #spree=
    # f = torch.zeros((W,H),device = device)
    context_input = torch.abs(torch.randn((cortical_delay,W,H), device = device)) # gamma
    # context_input /= context_input.sum(0) #
    new_context_input = torch.abs(torch.randn((cortical_delay, W, H),device = device))
    x_record = []
    # encoding_record = []

    for iter in range(iterations):
        x_record.append([])
        # encoding_record.append([])
        for t in range(cortical_delay):
            mem = x*context_input[t]
            # noise = torch.rand(x.size(),device = device)
            accumulate_input = mem  #+noise
            output,encoding,_ = net(accumulate_input.reshape(1,-1).type(dtype=torch.float32))
            new_context_input[t,:,:] = output.reshape(W,H)
            x_record[-1].append(np.array(accumulate_input.detach().cpu()))

        context_input = new_context_input
    x_record = np.array(x_record)

    return x_record


def clustering_on_each_datset(net_name,data_name,para=[32,9],back=10,smooth=2):
    net = torch.load('../../tmp_net/'+net_name)
    _, multi, _, _, multi_label, _ = gain_dataset("../../tmp_data", data_name)
    print(multi[0].shape, multi_label[0])
    AMI_mean = []
    AMI_score_5 = []
    # seed_list = np.random.randint(1,1000,5)
    seed_list = [111,222,333,444,555]
    for seed in seed_list: # different random seed
        setup_seed_pytorch(seed)
        AMI_score = []
        idx_dict = {}
        for i in range(6000):
            spk = DAE_clustering(np.array(multi[i], dtype=np.float32), net,cortical_delay=para[0])
            pred_low = k_means(spk,multi_label[i],show=False, back=back, smooth=smooth)
            ami_score = evaluate_grouping(multi_label[i], pred_low)
            print(data_name," seed:", seed, ' id:', i, " AMI score:", ami_score)
            idx_dict[ami_score] = i
            AMI_score.append(ami_score)
        AMI_score.sort()
        AMI_mean.append(np.array(AMI_score).mean())
        AMI_score_5.append(AMI_score)
    return np.array(AMI_score_5), np.array(AMI_mean)

# 'mnist_shape_bcdae_net_0.6.pty'
# 'mnist_shape_cnn3.pty
net_name_list=['bars_bae_net.pty','corners_bae_net.pty','shapes_dae_0.8_400_net2.pty','multi_mnist_bcdae_net_0.9.pty','mnist_shape_bae_net_bast_net_para2.pty']
data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = [[54,6],[28,8],[32,9],[29,12],[29,9]]
start_time = datetime.datetime.now()
for i in range(5):
    print("i = ", i)
    AMI_score, AMI_mean = clustering_on_each_datset(net_name_list[i],data_name_list[i],para_list[i])
    np.save('./AMI_npys/ami_DAE_score_5array_'+data_name_list[i],AMI_score)
    np.save('./AMI_npys/ami_DAE_score_5mean_' + data_name_list[i], AMI_mean)
    print(data_name_list[i]+' :  ',AMI_score.shape)
end_time = datetime.datetime.now()
print("time used:", end_time-start_time)




