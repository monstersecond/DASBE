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

def setup_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

Vth = 1+0.00001


color_map = ['goldenrod','steelblue','olivedrab']

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

def load():
    results = np.load('./results.npy',allow_pickle=True)
    results = results.ravel()[0]
    syn = np.array(results['syn_score'])
    rate = np.array(results['rate_score'])
    km = np.array(results['km_score'])
    color = np.array(results['color'])
    corr = np.array(results['corr'])
    spk = np.array(results['spk'])
    groups = np.array(results['groups'])
    selected = np.array(results['selected'])
    print(syn.shape,rate.shape,color.shape, corr.shape)
    syn_rate_figdata=[]
    data_len = syn.shape[1]
    for t in range(data_len):
        for i in range(5):
            # syn_rate_figdata.append([32 * t, syn[i, t], rate[i, t]])
            syn_rate_figdata.append([32 * t, syn[i, t], 'synchrony'])
            syn_rate_figdata.append([32 * t, rate[i, t], 'rate'])
    # df = pd.DataFrame(syn_rate_figdata, columns=['t', 'synchrony score', 'rate score'])
    df_score = pd.DataFrame(syn_rate_figdata, columns=['t', 'score','type'])

    corr_fig = []
    corr_len = (corr.shape[2]+1)/2
    t_axis = np.arange(-corr_len + 1, corr_len, 1)
    shape_list = ['group 1','group 2', 'group 3']
    for t in range(corr.shape[2]):
        for s in range(corr.shape[0]):
            # corr_fig.append([t_axis[t], corr[s, 0, t], corr[s, 1, t], corr[s, 2, t]])
            for i in range(3):
                corr_fig.append([t_axis[t],corr[s,i,t], shape_list[i]])
    df_corr = pd.DataFrame(corr_fig, columns=['t_axis', 'auto-correlation','label'])
    # df_corr = pd.DataFrame(corr_fig, columns=['t_axis', 'auto-correlation 1', 'auto-correlation 2', 'auto-correlation 3'])

    print(spk.shape,groups.shape,selected.shape)
    SELECTED = []
    num = selected.shape[3]
    delay = selected.shape[2]
    for t in range(delay):  # 64
        for i in range(3):  # label group
            for j in range(3): # selected id
                SELECTED.append([t,selected[0,i,t,j],shape_list[i],shape_list[i]+str(j)])
        #     SELECTED.append([t, groups[0, i, t],shape_list[i],shape_list[i]+'group'])
        # SELECTED.append([t, groups[0, :, t].sum(), 'total', 'total'])
    ds = pd.DataFrame(SELECTED, columns=['t', 'spikes','group','idx'])

    GR=[]
    for t in range(delay):
        for i in range(3):
            GR.append([t, groups[0, i, t], shape_list[i]])
        GR.append([t, groups[0, :, t].sum(), 'total'])
    gr = pd.DataFrame(GR, columns=['t', 'spike count','group'])

    print("spk",spk.shape)

    k_s_r = []
    km_rate = []
    print("km",km.shape)
    for i in range(km.shape[0]):
        for it in range(km.shape[1]):
            # k_s_r.append([km[i,it],syn[i,it],rate[i,it],it])
            k_s_r.append([km[i, it], syn[i, it], 'synchrony score', it])
            k_s_r.append([km[i, it], rate[i, it], 'rate score', it])
    # ksr = pd.DataFrame(k_s_r,columns=["Silhouette score","Synchrony score","Rate score","iteration step"0arsx])
    ksr = pd.DataFrame(k_s_r, columns=["silhouette score", "synchrony or rate score", "y_axis", "iteration step"])


    return df_score, df_corr, ds, gr, color,spk, ksr
def draw_score(data):
    print("drawing")
    # plt.figure()
    # plt.subplot(2,1,1)
    # fig = sns.lineplot(x="t", y="synchrony score", data=data, palette='flare')
    # fig.set_ylim(-0.6,1)
    # plt.subplot(2, 1, 2)
    # fig = sns.lineplot(x="t", y="rate score", data=data, palette='flare')
    # fig.set_ylim(-0.6, 1)
    # for font in font_manager.fontManager.ttflist:
    #     print(font)
    # fig = sns.lineplot(x="t", y="score", data=data, hue='type',style='type',size='type') #palette='flare'
    line_fig = sns.relplot(x="t", y="score", data=data, hue='type', style='type', size='type', kind="line", aspect=4)  # palette='flare'
    line_fig.ax.axhline(0,color='grey')
    leg=line_fig._legend
    leg.set_bbox_to_anchor([0.9,0.65]) # position of legend
    line_fig.set_ylabels('score')
    # plt.xlabel("t", fontsize=13)
    # plt.ylabel("score", fontsize=13)
    # plt.legend(prop={"size": 15})
    # fig.set_yticks([-0.2,0,0.2,0.8,1])
    # fig.legend(fontsize=12)
    # fig.set_xticklabels(fig.get_xticklabels(),rotation=-90)
    # fig.set_yticklabels(fig.get_xticklabels(), rotation=-90)
    # line_fig = fig.get_figure()
    plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['figure.dpi'] = 1000
    line_fig.savefig("./syn_rate_figure")
    plt.show()

def draw_corr(corr):
    # fig = sns.lineplot(x="t_axis", y="auto-correlation", data=corr, hue='group label') #palette='flare'
    fig = sns.relplot(x="t_axis", y="auto-correlation", data=corr, hue = 'label', row='label', kind='line', palette=color_map,legend=False)  # palette='flare'
    fig.set(yticks=[0,1])
    fig.set_titles("")
    fig.set_ylabels("")
    fig.set_xlabels("")
    # leg = fig._legend
    # leg.set_bbox_to_anchor([0.9, 0.65])  # position of legend

    # F = plt.figure()
    # plt.subplot(3,1,1)
    # themes=sns.xkcd_palette(['gold'])
    # fig = sns.lineplot(x="t_axis", y="auto-correlation 1", data=corr, palette=themes)
    # fig.legend(fontsize=12)
    # plt.subplot(3, 1, 2)
    # themes = sns.xkcd_palette(['red'])
    # fig = sns.lineplot(x="t_axis", y="auto-correlation 2", data=corr, palette=themes)
    # fig.legend(fontsize=12)
    # plt.subplot(3, 1, 3)
    # themes = sns.xkcd_palette(['green'])
    # fig = sns.lineplot(x="t_axis", y="auto-correlation 3", data=corr, palette=themes)
    # fig.legend(fontsize=12)
    # F.savefig("./corr_figure", dpi=400)
    plt.show()
    # line_fig = fig.getfigure()
    # line_fig.savefig("./corr_figure", dpi=400)
    plt.rcParams['savefig.dpi'] = 1000
    plt.rcParams['figure.dpi'] = 1000
    fig.savefig("./corr_figure")

def draw_spk(data, gr):
    #draw selected spikes
    sns.set_theme(style='white')
    sns.set(font_scale=0)
    g = sns.FacetGrid(data,row='idx',hue='group', sharex=True, sharey=True,despine=True, margin_titles=True)
    g.despine(left=True,bottom=True)
    g.set(xticks=[])
    g.set(yticks=[])
    # g.set(axis='off')
    # plt.axis('off')
    g.map(sns.barplot,"t","spikes")
    g.set_axis_labels('')
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.figure.subplots_adjust(top=0.8,bottom=0.2,left=0.1,right=0.7)
    g.set(xticks=[],yticks=[])
    # g.set_xticklabels(labels=[])
    # g.set_yticklabels(labels=[])
    # plt.axis('off')
    plt.show()

    #draw grouped spike counts
    sns.set_theme(style='white')
    sns.set(font_scale=0)
    g = sns.FacetGrid(gr, row='group', hue='group', despine=True, margin_titles=True)
    g.despine(left=True, bottom=True)
    g.set(xticks=[])
    g.set(yticks=[])
    g.map(sns.barplot, "t", "spike count")
    g.set_axis_labels('')
    g.figure.subplots_adjust(wspace=0, hspace=0)
    g.figure.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.7)
    g.set(xticks=[], yticks=[])
    # g.set_xticklabels(labels=[])
    # g.set_yticklabels(labels=[])
    # plt.axis('off')
    plt.show()

def draw_coloring(color):
    color = color[0,29,0,:,:,:].reshape(-1,3) #(28X28)X3
    print(color.shape)
    plt.figure()
    plt.subplot(3,1,1)
    print(color[:,0].shape)
    plt.bar(np.arange(784),color[:,0])
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(784),color[:,1])
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(784),color[:,2])
    plt.show()
    return

from matplotlib.colors import hsv_to_rgb
def coloring(spk, para=[32,9],i=0, idx = [0,2,4,9,19,29], show=True):
    spk = spk[i]
    iteration = spk.shape[0]
    delay = spk.shape[1]
    # print('spk_shape:', spk.shape) # iter X delay X W X H ?
    spk = spk.transpose(0,2,3,1)
    # print('spk_shape:',spk.shape) # # iter X W X H X delay ?
    T = spk.shape[-1]
    # print("T=",T)
    w = para[0]/para[1]
    colors = 0.5*(np.sin(2*w*np.pi*np.linspace(0, 1, T, endpoint=False))+1) # [0,1] and periodic
    colors = np.tile(colors,(spk.shape[0],spk.shape[1], spk.shape[2],1))
    #print('colors:', colors)
    # colors = colors.transpose(1,2,3,0)
    # print('colors_shape:', colors.shape)

    results = spk*colors
    palette = sns.color_palette()
    # sns.palplot(palette)
    if results.shape[-1] != 3:
        nr_colors = results.shape[-1]
        hsv_colors = np.ones((nr_colors, 3))
        hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2 / 3) % 1.0
        color_conv = hsv_to_rgb(hsv_colors)
        results = results.reshape(-1, nr_colors).dot(color_conv).reshape(results.shape[:-1] + (3,))
    if show:
        plt.figure()
        # for i in range(len(idx)):
        #     plt.subplot(len(idx),1,i+1)
        #     plt.imshow(results[idx[i], :,:,:], interpolation='nearest')
        #     plt.axis('off')
        #     plt.savefig('coloring_'+str(idx[i])+".png")
        # plt.show()
        for i in range(len(idx)):
            plt.imshow(results[idx[i], :, :, :], interpolation='nearest')
            plt.axis('off')
            plt.rcParams['savefig.dpi'] = 1000
            plt.rcParams['figure.dpi'] = 1000
            plt.savefig('coloring_'+str(idx[i])+".png", dpi=500)

    return results

def draw_ksr(ksr):
    # fig1 = sns.lmplot(x="Silhouette score", y="Synchrony score",data=ksr)  # palette='flare'
    # fig2 = sns.lmplot(x="Silhouette score", y="Rate score", data=ksr)
    fig = sns.lmplot(x="silhouette score", y="synchrony or rate score", row='y_axis',hue = 'y_axis', markers=['.','.'], data=ksr, aspect=1.5, sharey=False)
    fig.set_titles("")
    fig.set_xlabels("clustering score")
    fig.set_ylabels("")
    fig.set(yticks=[0, 1])
    ax = fig.facet_axis(0, 0)
    ax.set_ylabel("synchrony score", loc='center')
    # ax.set_ylim(0,1)
    ax = fig.facet_axis(1, 0)
    # ax.set_ylim(-1, 1)
    ax.set_ylabel("rate score")
    # axes = fig.axes
    # axes[0,0].set_ylim(0,)
    # axes[1,0].set_ylim(-0.5,)
    # plt.hlines(0,0,32*30, color='grey')
    # ax = fig.facet_axis(0,0)
    # leg = fig._legend
    # print(leg)
    # leg.set_bbox_to_anchor([0.9, 0.65])  # position of legend
    # ax.hlines(0,0,1, color='lightgrey',linewidth=1)
    # leg = fig._legend
    # leg.set_bbox_to_anchor([0.9, 0.65])  # position of legend
    # fig.set_ylabels('score')
    plt.rcParams['savefig.dpi']=1000
    plt.rcParams['figure.dpi']=1000
    fig.savefig("./ksr_figure")
    plt.show()



pd.options.display.notebook_repr_html=False
# plt.rcParams['Figure.dpi']=75
sns.set_theme(style='darkgrid')
data, corr, spk_selected, gr, color,spk,ksr = load()
draw_score(data)
draw_corr(corr)
coloring(spk)
draw_ksr(ksr)