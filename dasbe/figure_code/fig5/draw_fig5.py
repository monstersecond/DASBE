import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.font_manager as font_manager
import sys
sys.path.append("../../")

from dataset_utils import gain_dataset
from FCA import HC
from analysis import labeled_synchrony_measure, fMRI_measure, k_means,decode_with_kmeanMask, evaluate_grouping, autocorrelation, coloring, k_means_var, victor_purpura_disssimilarity, draw_spikes, draw_context,LFP, explore_timescale, DBSCAN_cluster, VP_silhouette, step_wise_analysis, feature_neuron, find_timescale_with_silhouette
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

plt.rcParams['savefig.dpi'] = 1000 #图片像素 
plt.rcParams['figure.dpi'] = 1000 #分辨率

def setup_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

Vth = 1+0.00001

data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = [[54,6],[28,8],[32,9],[29,12],[28,8]]

def load():
    print("loading npy")
    ami_syn_rate_compare={}
    syn_score = {}
    rate_score = {}
    km_score= {}
    inertia={}
    for i in range(5):
        name = data_name_list[i]
        ami_syn_rate_compare[name] = np.load('../../table1/unified_clustering/AMI_npys/syn_rate_AMI_pairs'+name + '.npy')
        syn_score[name] = np.load('../../table1/AMI_npys/syn_score_all_' + name +'.npy')
        rate_score[name] = np.load('../../table1/AMI_npys/rate_score_all_' + name + '.npy')
        km_score[name] = np.load('../../table1/AMI_npys/km_score_all_' + name + '.npy')
        # inertia[name] = np.load('../../table1/AMI_npys/Inertia_all_' + name + '.npy')
        ami_syn_rate_compare[name] = ami_syn_rate_compare[name].squeeze()
        # print(ami_syn_rate_compare.shape)
        # print(syn_score.shape)
        # print(rate_score.shape)
    scatter_data1 = []
    scatter_data2 = []
    for name in data_name_list:
        for i in range(ami_syn_rate_compare[name].shape[0]):
            scatter_data1.append([ami_syn_rate_compare[name][i,0],ami_syn_rate_compare[name][i,1],name])
            scatter_data2.append([ami_syn_rate_compare[name][i, 0], ami_syn_rate_compare[name][i, 2], name])
    sd1 = pd.DataFrame(scatter_data1,columns=['AMI score', 'synchrony score','dataset'])
    sd2 = pd.DataFrame(scatter_data2, columns=['AMI score', 'rate score', 'dataset'])

    syn_Scores = []
    rate_Scores = []
    for j in range(5):
        name = data_name_list[j]
        for i in range(syn_score[name].shape[0]):
            for t in range(syn_score[name].shape[1]):
                syn_Scores.append([t,syn_score[name][i,t],name])
                rate_Scores.append([t, rate_score[name][i, t], name])
    ss = pd.DataFrame(syn_Scores,columns=['iteration steps', 'synchrony score','dataset'])
    rs = pd.DataFrame(rate_Scores, columns=['iteration steps', 'rate score', 'dataset'])

    kisr = []

    for j in range(5):
        name = data_name_list[j]
        for i in range(km_score[name].shape[0]):
            for t in range(km_score[name].shape[1]):
                # kisr.append([km_score[name][i,t],inertia[name][i, t], syn_score[name][i,t] ,rate_score[name][i, t], name])
                dice = np.random.rand() # sample is too many
                if dice<1/30:
                    kisr.append(
                        [km_score[name][i, t], [], syn_score[name][i, t], rate_score[name][i, t], name])

    Kisr = pd.DataFrame(kisr,columns=['km score', 'inertia','syn score','rate score','dataset'])
    return sd1,sd2, ss,rs, Kisr

def draw_scatter(sd1,sd2):
    fig1 = sns.lmplot('synchrony score','AMI score', hue='dataset', robust=True,markers=['.','x','*','s','v'], scatter_kws={"s":5}, data=sd1)
    fig2 = sns.lmplot('rate score', 'AMI score', hue='dataset', robust=True, markers=['.','x','*','s','v'],scatter_kws={"s":5},data=sd2)
    fig1.savefig("./syn_ami_figure", dpi=400)
    fig2.savefig("./rate_ami_figure", dpi=400)
    plt.show()

def draw_scores(ss,rs):
    fig1 = sns.lineplot(x="iteration steps", y="synchrony score", data=ss, hue='dataset', style='dataset', size='dataset')
    fig1.set_yticks(ticks=[-0.5,0,0.5,1])
    fig1.set_xticks(ticks=[0, 10, 20, 30])
    plt.tick_params(labelsize=14)
    plt.xlabel("Iteration steps",fontsize=14)
    plt.ylabel("Synchrony score",fontsize=14)
    plt.legend(prop={"size":14})
    fig1.set(ylim=(-0.5,1))
    line_fig1 = fig1.get_figure()
    line_fig1.savefig("./syn_score_all_figure", bbox_inches='tight')
    plt.show()
    fig2 = sns.lineplot(x="iteration steps", y="rate score", data=rs, hue='dataset', style='dataset', size='dataset')
    fig2.set_yticks(ticks=[-0.5, 0, 0.5, 1])
    fig2.set_xticks(ticks=[0, 10, 20, 30])
    plt.tick_params(labelsize=14)
    plt.xlabel("Iteration steps", fontsize=14)
    plt.ylabel("Rate score", fontsize=14)
    plt.legend(prop={"size": 14})
    fig2.set(ylim=(-0.5,1))
    line_fig2 = fig2.get_figure()
    line_fig2.savefig("./rate_score_all_figure",bbox_inches='tight') #dpi=
    plt.show()

def draw_ksir(ksir):
    fig1 = sns.lmplot('km score','syn score',hue='dataset',markers='.',scatter_kws={'s':4}, data=ksir,legend=False)
    # plt.ylim(-1,1)
    # fig1.set_yticks(ticks=[0, 0.5, 1])
    # fig1.set_xticks(ticks=[0, 0.5, 1])
    plt.tick_params(labelsize=14)
    plt.xticks([0.2,0.4,0.6,0.8,1])
    plt.yticks([0.2,0.4,0.6,0.8,1])
    plt.xlabel("Clustering score", fontsize=14)
    plt.ylabel("Synchrony score", fontsize=14)
    plt.legend(prop={"size": 12})
    fig2 = sns.lmplot('km score', 'rate score', hue='dataset', markers='.', scatter_kws={'s':4}, data=ksir, legend=False)
    # plt.ylim(-1, 1)
    plt.tick_params(labelsize=14)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([-1,-0.5, 0, 0.5, 1])
    plt.xlabel("Clustering score", fontsize=14)
    plt.ylabel("Rate score", fontsize=14)
    plt.legend(prop={"size": 12})

    # fig3 = sns.lmplot('inertia', 'syn_score', hue='dataset', data=ksir)
    # fig4 = sns.lmplot('inertia', 'rate_score', hue='dataset', data=ksir)
    plt.show()
    fig1.savefig('./ksir_syn.png',bbox_inches='tight')
    fig2.savefig('./ksir_rate.png',bbox_inches='tight')





pd.options.display.notebook_repr_html=False
# plt.rcParams['Figure.dpi']=75
sns.set_theme(style='darkgrid')
sd1,sd2,ss,rs, ksir = load()
# draw_scatter(sd1,sd2)
draw_scores(ss,rs)
draw_ksir(ksir)