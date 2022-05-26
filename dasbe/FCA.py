import random
from sklearn import cluster
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from dataset_utils import gain_dataset,gain_verf_dataset
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def closest_spike(s1, s2):
    f1 = np.where(s1 > 0)[0]
    f2 = np.where(s2 > 0)[0]
    if len(f1) == 0 and len(f2) == 0:
        return 0
    elif len(f1) == 0 or len(f2) == 0:
        return len(s1)
    f1 = np.expand_dims(f1,1)
    f2 = np.expand_dims(f2,0)
    dt1 = np.min(np.abs(f1-f2), axis=1).sum()
    return dt1


def AMD(s1, s2):
    T = s1.shape[0]
    dt1 = closest_spike(s1,s2)
    dt2 = closest_spike(s2,s1)
    N1 = s1.sum()
    N2 = s2.sum()
    D12 = dt1*((N2+1)/(T*(N1+1)))
    D21 = dt2*((N1+1)/(T*(N2+1)))
    O12 = (D12+D21)/2
    return O12


def victor_purpura_metric(s1, s2, q=0.1):
    f1 = np.where(s1 > 0)[0]
    f2 = np.where(s2 > 0)[0]
    f1_len = len(f1)
    f2_len = len(f2)
    f = np.zeros((f1_len + 1, f2_len + 1))
    for i in range(f1_len + 1):
        f[i, 0] = i
    for j in range(f2_len + 1):
        f[0, j] = j
    for i in range(1, f1_len + 1):
        for j in range(1, f2_len + 1):
            f[i, j] = f[i-1, j-1] + abs(f1[i-1] - f2[j-1]) * q
            f[i, j] = min(f[i, j], f[i, j-1] + 1)
            f[i, j] = min(f[i, j], f[i-1, j] + 1)
    return f[f1_len, f2_len]


def AMD_similarity(spk, no_back_ground = True):
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)

    spk = spk.reshape(spk.shape[0], -1)

    if no_back_ground==True:
        tmp_idx = np.where(spk.sum(0)>0)[0]
        spk = spk[:,tmp_idx]
    print("spk_AMD:  ", spk.shape)
    amd = np.zeros((spk.shape[1],spk.shape[1]))
    print("amd: ",amd.shape)
    for i in range(spk.shape[1]):
        for j in range(spk.shape[1]):
            amd[i,j] = AMD(spk[:,i], spk[:,j])
    print("AMD_SHAPE: ", amd.shape)
    plt.figure()
    plt.imshow(amd / np.max(amd))
    plt.show()
    return amd


def AMD_similarity_(spk, no_back_ground = True):
    if no_back_ground==True:
        tmp_idx = np.where(spk.sum(0)>0)[0]
        spk = spk[:,tmp_idx]
    print("spk_AMD:  ", spk.shape)
    amd = np.zeros((spk.shape[1],spk.shape[1]))
    print("amd: ",amd.shape)
    for i in range(spk.shape[1]):
        for j in range(spk.shape[1]):
            amd[i,j] = victor_purpura_metric(spk[:,i],spk[:,j])
    print("AMD_SHAPE: ", amd.shape)
    return amd


def HC(spk, no_background=False, target_cluster_num=3, visible=True, debug=True): 
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    spk = spk.reshape(spk.shape[0], -1)

    time_step = spk.shape[0]  
    neuron_num = spk.shape[1]  

    if no_background == True:
        tmp_idx = np.where(spk.sum(0) > 0)[0]
        spk = spk[:, tmp_idx]
    
    amd = AMD_similarity_(spk, no_background)    
    idx = list(range(0, neuron_num))             
    cluster_res = [[i] for i in range(neuron_num)]
    amd[range(neuron_num), range(neuron_num)] = np.inf

    if visible:
        plt.figure()
        plt.ion()
        img = np.zeros((sizex*sizey, ))
        cls = 0
        for i in idx:
            img[cluster_res[i]] = cls
            cls += 1
        plt.imshow(img.reshape(sizex, sizey))
        plt.pause(0.1)

    cur_cluster_num = neuron_num
    while cur_cluster_num > target_cluster_num:
        if debug:
            print("cur_cluster_num:", cur_cluster_num)
        tmp_idx = np.argmin(amd[idx][:, idx])
        tmp_idx1 = tmp_idx // cur_cluster_num
        tmp_idx2 = tmp_idx % cur_cluster_num
        idx1 = idx[tmp_idx1]
        idx2 = idx[tmp_idx2]
        if debug:
            print(idx1, idx2)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        cluster_res[idx1].extend(cluster_res[idx2])

        
        
        if random.random() >= 0.5:
            spk[:, idx1] = spk[:, idx2]
        idx.remove(idx2)

        for i in idx:   
            amd[idx1, i] = amd[i, idx1] = victor_purpura_metric(spk[:, idx1], spk[:, i])
        amd[idx1, idx1] = np.inf

        cur_cluster_num -= 1

        if visible:
            plt.cla()
            img = np.zeros((sizex*sizey, ))
            cls = 0
            for i in idx:
                img[cluster_res[i]] = cls
                cls += 1
            plt.imshow(img.reshape(sizex, sizey))
            plt.pause(0.1)

    if visible:
        plt.ioff()
        plt.show()

    if debug:
        print(cluster_res)
        for i in idx:
            print(len(cluster_res[i]))
            print(cluster_res[i])

    img = np.zeros((sizex*sizey, ))
    cls = 0
    for i in idx:
        img[cluster_res[i]] = cls
        cls += 1
    
    if visible:
        plt.figure()
        plt.imshow(img.reshape(sizex, sizey))
        plt.show()
    
    return img.reshape(sizex, sizey)


if __name__ == "__main__":
    s1 = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    s2 = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    

    data = np.random.randint(0, 2, (5, 2, 5, 5))
    print(data.shape)
    print(HC(data, debug=False, visible=False))

