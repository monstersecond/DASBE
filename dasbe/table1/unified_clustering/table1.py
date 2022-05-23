import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
sys.path.append('../../')

from dataset_utils import gain_dataset
from FCA import HC
from analysis import labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

Vth = 1+0.00001

net_name_list=['bars_bae_net.pty','corners_bae_net.pty','shapes_dae_0.8_400_net.pty','multi_mnist_bcdae_net_0.9.pty','mnist_shape_bae_net_bast_net_para2.pty']
# data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = [[54,6],[28,8],[32,9],[29,12],[29,9]]
# model_name_list = ['HNN','DAE','pcnn']
model_name_list = ['HNN','DAE','pcnn']

def setup_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def load_data():
    print("loading npy")
    data = {}
    for name in data_name_list:
        data[name] = {}
        for model in model_name_list:
            ami = np.load('./AMI_npys/AMI_score_saved/ami_' + model + '_score_5mean_' + name + '.npy')
            print(ami.shape)
            data[name][model]=ami
    print("load finish")
    return data

def mean_stddev(data):
    print("mean_stddev")
    mean={}
    stddev={}
    for name in data_name_list:
        mean[name] = {}
        stddev[name] = {}
        for model in model_name_list:
            mean[name][model]=data[name][model].mean()
            stddev[name][model] = data[name][model].std()
            print(name,model,mean[name][model],stddev[name][model])
    np.save('./table1.npy',[mean,stddev])

data=load_data()
mean_stddev(data)
