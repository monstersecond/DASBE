import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import copy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from dataset_utils import gain_dataset, gain_verf_dataset

from FCA import victor_purpura_metric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_spikes(spike_record, img, name = '', back_iter=10):
    iter = spike_record.shape[0]
    T = spike_record.shape[1]
    plt.figure()
    for i in range(iter-back_iter, iter):
        for j in range(T):
            plt.subplot(back_iter+1, T, (back_iter-(iter-i))*T+j+1)
            plt.imshow(spike_record[i,j,:,:])
            plt.axis('off')
    plt.subplot(back_iter + 1, T, back_iter*T + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('./'+name+'spike_record.png')
    plt.show()


def draw_spikes_dynamic(spike_record, img, name='', back_iter=10):
    iter = spike_record.shape[0]
    T = spike_record.shape[1]
    plt.figure()
    for i in range(iter-back_iter, iter):
        for j in range(T + 1):
            plt.subplot(back_iter, T + 1, (back_iter - (iter - i)) * (T + 1) + j + 1)
            if j < T:
                plt.imshow(spike_record[i, j, :, :])
            else:
                plt.imshow(img[i])
            plt.axis('off')
    plt.savefig('./' + name + 'spike_record.png')
    plt.show()


def draw_spikes_dynamic_2(spike_record, img, name='', rng=[0, 1, 2, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]):
    iter = spike_record.shape[0]
    T = spike_record.shape[1]
    cur_fig_id = 1
    plt.figure(figsize=(24, 16))
    for i in rng:
        for j in range(T + 1):
            plt.subplot(len(rng), T + 1, cur_fig_id)
            if j < T:
                colors = ['black', 'yellow'] 
                cmap = mpl.colors.ListedColormap(colors)
                plt.imshow(spike_record[i, j, :, :], cmap=cmap)
            else:
                colors = ['white', 'gold', 'royalblue', 'limegreen'] 
                shape_num = img[i].max() + 1
                cmap_obj = mpl.colors.ListedColormap(colors[:int(shape_num)])
                plt.imshow(img[i], cmap=cmap_obj)
            plt.axis('off')
            cur_fig_id += 1
    plt.savefig('./'+name+'spike_record.png')
    plt.show()


def draw_context(context_record, img,name='', back_iter=10):
    iter = context_record.shape[0]
    T = context_record.shape[1]
    plt.figure()
    for i in range(iter-back_iter, iter):
        for j in range(T):
            plt.subplot(back_iter+1,T,(back_iter-(iter-i))*T+j+1)
            plt.imshow(context_record[i,j,:,:])
            plt.axis('off')
    plt.subplot(back_iter + 1, T, back_iter*T + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./'+name+'context_record.png')
    plt.show()


def factor_analysis(name='shapes'):
    net = torch.load('./tmp_net/' + name + '_net.pty')
    shapes = gain_verf_dataset("./tmp_data", name)
    a_encode = []
    print("dim of factors:  ",len(shapes))
    for types in range(len(shapes)):
        shape = torch.tensor(shapes[types], device=device)
        a_encode.append(np.zeros_like(np.array(net.mem.detach().cpu())))
        print(shape.shape, a_encode[-1].shape)
        plt.figure()
        plt.imshow(shapes[types][0,:,:])
        plt.show()
        for id in range(shape.shape[0]):
            print(id, shape[id,:,:].shape)
            _,encoding,_ = net(shape[id,:,:].reshape(1,-1).type(dtype=torch.float32))
            a_encode[-1] += np.array(encoding.cpu().detach().squeeze())
        a_encode[-1] /= shape.shape[0]

    plt.figure()
    for types in range(len(shapes)):
        plt.subplot(1,3,types+1)
        print(a_encode[types])
        plt.imshow(a_encode[types].reshape(16,16))
    plt.savefig('./shape_hidden_layer analysis.png')
    plt.show()


def rate(spk,win,back=-1): # not necessary
    if back!=-1:
        spk = spk[-back:, :, :, :]
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    T = spk.shape[0]
    rate = np.zeros((T,sizex,sizey))
    for t in range(win,T):
        if t<win:
            rate[t,:,:] = spk[:t,:,:].mean(0)
        else:
            rate[t,:,:] = spk[t-win:t,:,:].mean(0)
    rate = rate.reshape(T,-1)
    plt.figure()
    for i in range(rate.shape[1]):
        plt.plot(rate[:,i])
    plt.show()


def LFP(context):  # local field potential
    lfp = context.mean((2, 3))
    lfp = lfp.reshape(-1)
    plt.figure()
    plt.plot(np.arange(len(lfp)),lfp)
    plt.show()


def labeled_synchrony_measure(spk,label, back = 2, num = 2, show=True): # low level only
    K = np.max(label)
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk[-back:,:,:,:]
    spk = spk.reshape(-1,sizex,sizey)
    T = spk.shape[0]
    name = ['red','blue','green','yellow','purple','orange','pink','silver','snow','teal','navy','gray']
    print("K: ",K)
    groups = np.zeros((int(K),spk.shape[0],spk.shape[1],spk.shape[2])) # K,T,W,H
    selected = np.zeros((int(K),spk.shape[0],num)) # K,T,num

    for t in range(spk.shape[0]):
        for idx in range(spk.shape[1]):
            for idy in range(spk.shape[2]):
                if spk[t,idx,idy]>0:
                    k = int(label[idx,idy])
                    groups[k-1,t,idx,idy]=1

    for k in range(int(K)):
        id_pol = np.where(groups.sum(1)[k,:,:]>0)
        idx_tmp = np.random.choice(np.arange(len(id_pol[0])),size = num, replace=False)
        selected[k,:,:] = groups[k,:,id_pol[0][idx_tmp],id_pol[1][idx_tmp]].T

    groups = groups.sum(axis=(2,3))

    if show == True:
        t_axis = np.arange(1,T+1)
        plt.figure()
        plt.subplot(K+1+num*K,1,1)
        #print(groups[0,:])
        for k in range(int(K)):
            plt.bar(t_axis,groups[k,:],color=name[k])
        for k in range(int(K)):
            plt.subplot(K+1+num*K, 1, k+2)
            plt.bar(t_axis,groups[k,:],color = name[k])
        for k in range(int(K)):
            for i in range(num):
                plt.subplot(K + 1 + num * K, 1, K -1 + 2+k*num+i+1)
                plt.bar(t_axis, selected[k, :,i], color=name[k])
        plt.show()

    return groups, selected


def coloring(spk, para, back=10, show=True):
    iteration = spk.shape[0]
    delay = spk.shape[1]
    spk = spk.transpose(0,2,3,1)
    T = spk.shape[-1]
    w = para[0]/para[1]
    colors = 0.5*(np.sin(2*w*np.pi*np.linspace(0, 1, T, endpoint=False))+1) # [0,1] and periodic
    colors = np.tile(colors,(spk.shape[0],spk.shape[1], spk.shape[2],1))

    results = spk*colors
    palette = sns.color_palette()
    
    if results.shape[-1] != 3:
        nr_colors = results.shape[-1]
        hsv_colors = np.ones((nr_colors, 3))
        hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2 / 3) % 1.0
        color_conv = hsv_to_rgb(hsv_colors)
        results = results.reshape(-1, nr_colors).dot(color_conv).reshape(results.shape[:-1] + (3,))

    if show:
        plt.figure()
        for i in range(back):
            plt.subplot(back + 1,1,i + 1)
            plt.imshow(results[iteration-back+i, :,:,:], interpolation='nearest')
            plt.axis('off')
        plt.show()

    return results


def victor_purpura_disssimilarity(spk, GT, q, no_background=False, show = True):
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)

    spk = spk.reshape(spk.shape[0], -1)

    if no_background == True:
        # tmp_idx = np.where(spk.sum(0) > 0)[0]
        tmp_idx = np.where(GT.reshape(-1) > 0)[0]
        spk = spk[:, tmp_idx]

    vp = np.zeros((spk.shape[1], spk.shape[1]))

    for i in range(spk.shape[1]):
        for j in range(spk.shape[1]):
            vp[i, j] = victor_purpura_metric(spk[:, i], spk[:, j],q)

    new_idx = new_index(GT)
    vp1 = [[vp[i][j] for j in new_idx] for i in new_idx]
    vp = np.array(vp1)
    if show:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(vp / np.max(vp))
        plt.subplot(2, 1, 2)
        sim = 1 / (vp + 0.001)
        plt.imshow(sim)
        plt.show()
    else:
        return vp


def DBSCAN_cluster(spk,label,eps,min_n,q, back = 5,show=True):
    spk = spk[-back:, :, :, :]
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    spk = spk.reshape(spk.shape[0], -1)
    spk = spk.T  # (n_samples, n_featrues)
    estimator = DBSCAN(eps=eps, min_samples=min_n, metric = victor_purpura_metric,metric_params={'q':q})
    estimator.fit(spk)
    label_pred = estimator.labels_
    label_pred = label_pred.reshape(sizex, sizey)
    if show == True:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(label_pred)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.show()
    return label_pred


def new_index(GT,no_background=True):
    K = np.max(GT)
    GT = GT.reshape(-1)
    new_idx = []
    for i in range(int(K)+1):
        if no_background and i == 0:
            continue
        for j in range(len(GT)):
            if GT[j]==i:
                new_idx.append(j)
    new_idx = np.array(new_idx)
    if len(new_idx!=len(GT)) and not no_background:
        print("new_idx length not match!!!")
    return new_idx


def fMRI_measure(spk, GT, back = 10,width1=1, width2 = 1, alpha = 0.5, show=True): #both low & high level
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk[-back:,:,:,:]
    spk = spk.reshape(-1, sizex, sizey)
    spk = spk.reshape(spk.shape[0],-1)

    all_time = spk.shape[0]
    spk_pre = np.zeros_like(spk)
    spk_post = np.zeros_like(spk)
    for w in range(width1):
        spk_pre[0:all_time-w-1,:] = spk[w+1:,:]
    for w in range(width2):
        spk_post[w+1:,:] = spk[:all_time-w-1,:]

    fMRI = np.matmul(spk.T,spk) + alpha * np.matmul(spk.T,spk+spk_pre) + alpha* np.matmul(spk.T,spk_post)

    new_idx = new_index(GT)
    fMRI1 = [[fMRI[i][j] for j in new_idx] for i in new_idx]
    fMRI = np.array(fMRI1)
    if show:
        print("fMRI_SHAPE: ",fMRI.shape)
        plt.figure()
        plt.imshow(fMRI/np.max(fMRI))
        plt.show()
    return fMRI/np.max(fMRI)

def k_means(spk_to_copied,label,show = True, back = 10, smooth=0): #both low & high level
    spk = copy.deepcopy(spk_to_copied)
    spk = spk[-back:, :, :, :]
    alpha=1
    # note: update in 3.27
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    spk_tmp = copy.deepcopy(spk)

    for i in range(smooth-1):
        alpha *= 0.5
        spk_tmp[i+1:] += alpha * spk[:-i-1]

    spk = spk_tmp
    K = np.max(label)+1
    spk = spk.reshape(spk.shape[0], -1)
    spk = spk.T # (n_samples, n_featrues)
    estimator = KMeans(n_clusters=int(K), init='k-means++')
    estimator.fit(spk)
    label_pred = estimator.labels_
    label_pred = label_pred.reshape(sizex, sizey)

    if show == True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(label_pred)
        plt.subplot(1, 2, 2)
        plt.imshow(label)
        plt.show()

    return label_pred


def k_means_var(spk_to_be_copied, label, K, show=True, back=10, smooth=0): #both low & high level
    spk = copy.deepcopy(spk_to_be_copied)
    spk = spk[-back:, :, :, :]
    
    alpha=1
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    spk_tmp = copy.deepcopy(spk)

    for i in range(smooth-1):
        alpha*=0.5
        spk_tmp[i+1:] += alpha * spk[:-i-1]

    spk = spk_tmp
    spk = spk.reshape(spk.shape[0], -1)
    spk = spk.T # (n_samples, n_featrues)
    inertia = 10000000
    K_final = -1
    for k in K:
        estimator = KMeans(n_clusters=int(k),init = 'k-means++')
        estimator.fit(spk)
        label_pred_tmp = estimator.labels_
        label_pred_tmp = label_pred_tmp.reshape(sizex, sizey)
        inertia_tmp = estimator.inertia_
        if inertia_tmp < inertia:
            inertia = inertia_tmp
            label_pred = label_pred_tmp
            K_final = k
            print(k,inertia)

    if show == True:
        name = ['black', 'royalblue', 'limegreen', 'gold']
        cmap = mpl.colors.ListedColormap(name)
        plt.figure(figsize=(12, 24))
        plt.subplot(1, 2, 1)
        plt.imshow(label_pred, cmap=cmap)
        plt.axis('off')
        plt.box('off')
        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap=cmap)
        print("final_K = ",K_final, ';  inertia = ',inertia)
        
        plt.axis('off')
        plt.box('off')
        plt.show()
    return label_pred, label


import matplotlib as mpl


def decode_with_kmeanMask(net,enc,label_pred, img):
    print("decode_with_kmeanMask")
    print(label_pred.shape)
    K = np.max(label_pred) + 1
    plt.figure(figsize=(24, 8))
    name = ['coral', 'gold', 'royalblue', 'limegreen', 'coral', 'forestgreen', 'powderblue', 'red','blue','green','yellow','purple','orange','pink','silver','snow','teal','navy','gray']
    for k in range(K):
        mask = np.where(label_pred==k, 1, 0)
        x = torch.tensor(mask, device=device).reshape(-1).type(dtype=torch.float32)
        print("x:  ",x.size())
        Rimage = net.decoder(x)
        plt.subplot(1,K,k+1)
        colors = ['white', name[k]] 
        cmap = mpl.colors.ListedColormap(colors)
        plt.imshow(np.array(Rimage.reshape(28,28).cpu().detach()), cmap=cmap)
        plt.axis('on')
        plt.box('on')
        frame = plt.gca()
        
        frame.axes.get_yaxis().set_visible(False)
        frame.axes.get_xaxis().set_visible(False)

    plt.show()


def VP_silhouette(spk,labels,q,back = 10, no_background=True):
    spk = spk[-back:, :, :, :]
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    spk = spk.reshape(spk.shape[0], -1)
    labels = labels.reshape(-1)  # (n_samples)
    if no_background == True: # this make plot better!!!
        idx = np.where(labels.reshape(-1)!=0)[0]
        spk = spk[:,idx]
        labels = labels[idx]
    spk = spk.T  # (n_samples, n_featrues)
    print(spk.shape, labels.shape)
    score = silhouette_score(spk, labels, metric=victor_purpura_metric, q=q)

    return score


def syn_vs_rate_score(spk,labels,back = 10, no_background=True, smooth=2, step_wise=False):
    if not step_wise:
        spk = spk[-back:, :, :, :]
        sizex = spk.shape[2]
        sizey = spk.shape[3]
        spk = spk.reshape(-1, sizex, sizey)
    spk = spk.reshape(spk.shape[0], -1)
    labels = labels.reshape(-1)  # (n_samples)

    spk_tmp = copy.deepcopy(spk)
    alpha = 1
    for i in range(smooth - 1):
        alpha *= 0.5
        spk_tmp[i + 1:] += alpha * spk[:-i - 1]
    K = np.max(labels) + 1
    estimator = KMeans(n_clusters=int(K), init='k-means++')
    estimator.fit(spk_tmp.T)
    label_pred = estimator.labels_

    if no_background == True: # this make plot better!!!
        idx = np.where(labels.reshape(-1)!=0)[0]
        spk = spk[:,idx]
        spk_tmp1 = copy.deepcopy(spk_tmp)[:,idx]
        label_pred = label_pred[idx]
    spk = spk.T  # (n_samples, n_featrues)
    score_syn = silhouette_score(spk,label_pred,metric=victor_purpura_metric,q = 1/6)
    score_rate = silhouette_score(spk, label_pred, metric=victor_purpura_metric, q=0)
    score_k_means = silhouette_score(spk_tmp1.T, label_pred)
    inertia = estimator.inertia_
    return score_syn, score_rate, score_k_means, inertia


def explore_timescale(spk, GT, para, time_range=[], back = 10, no_back_ground=False): # another is to use VP silhouette as average VP_distance at tao
    spk = spk[-back:,:,:,:]
    T_max = 2*para[1]
    plt.figure()
    if no_back_ground:
        idx1 = np.where(GT!= 0)[0]
        idx2 = np.where(GT != 0)[1]
        spk = spk[:,:,idx1,idx2]
    q = 0
    vp = victor_purpura_disssimilarity(spk, GT, q, show=False)
    plt.subplot(4,1,1)
    plt.imshow(vp / np.max(vp))
    plt.axis('off')
    q = 1 / T_max
    vp = victor_purpura_disssimilarity(spk, GT, q, show=False)
    plt.subplot(4, 1, 2)
    plt.imshow(vp / np.max(vp))
    plt.axis('off')
    q = 1 / 3
    vp = victor_purpura_disssimilarity(spk, GT, q, show=False)
    plt.subplot(4, 1, 3)
    plt.imshow(vp / np.max(vp))
    plt.axis('off')
    q = 10000000
    vp = victor_purpura_disssimilarity(spk, GT, q, show=False)
    plt.subplot(4, 1, 4)
    plt.imshow(vp / np.max(vp))
    plt.axis('off')
    plt.show()


def find_timescale_with_silhouette(spk,GT,pred_label,time_range,back=10, no_bg=True):
    print('finding timescale of temporal binding')
    spk = spk[-back:,:,:,:]
    spk = spk.reshape(-1,spk.shape[2],spk.shape[3])
    print("spk.shape in find",spk.shape) # T,X,Y
    if no_bg:
        idx1 = np.where(GT != 0)[0]
        idx2 = np.where(GT != 0)[1]
        spk = spk[:, idx1, idx2]
        pred_label = pred_label[idx1,idx2]
        print("spk.shape in find", spk.shape)  # sample * feature
    spk = spk.T
    # sil plot with different tao
    sil_tao = []
    for tao in time_range:
        q = 1 / (tao + 0.00001)
        score = silhouette_score(spk, pred_label, metric=victor_purpura_metric, q=q)
        sil_tao.append(score)
    sil_tao = np.array(sil_tao)
    plt.figure()
    plt.plot(np.arange(len(sil_tao)),sil_tao)
    plt.figure()
    plt.bar(np.arange(len(sil_tao)),sil_tao)
    plt.show()
    print(time_range[np.argmax(sil_tao)])
    return sil_tao


def evaluate_grouping(true_groups, predicted):
    idxs = np.where(true_groups!=0.0)
    idcs = np.where(true_groups==0.0)
    tmp_tg = copy.deepcopy(predicted)
    tmp_tg[idcs]=-1
    score = adjusted_mutual_info_score(true_groups[idxs],predicted[idxs])

    return score

def autocorrelation(spk, label, corr_len = 100, show=True):
    K = np.max(label)
    sizex = spk.shape[2]
    sizey = spk.shape[3]
    spk = spk.reshape(-1, sizex, sizey)
    T = spk.shape[0]
    groups = np.zeros((int(K), spk.shape[0], spk.shape[1], spk.shape[2]))  # K,T,W*H
    for t in range(spk.shape[0]):
        for idx in range(spk.shape[1]):
            for idy in range(spk.shape[2]):
                if spk[t, idx, idy] > 0:
                    k = int(label[idx, idy])
                    groups[k - 1, t, idx, idy] = 1

    groups = groups.sum(axis=(2, 3))

    corr_record = np.zeros((int(K),2*corr_len-1))
    for i in range(corr_len):
        for k in range(int(K)):
            peak = np.dot(groups[k, :], groups[k, :])
            if i==0:
                # corr = np.dot(groups[k,:], groups[k,:])
                corr=1
            else:
                #print(i, groups.shape, groups[k,:-i].shape, groups[k,i:].shape)
                #corr = groups[k,:-i]*groups[k,i:]

                # corr = np.dot(groups[k,:-i],groups[k,i:])
                corr = np.dot(groups[k, :-i], groups[k, i:])/peak
            corr_record[k,corr_len-1-i] = corr
            corr_record[k, corr_len-1 + i] = corr

    t_axis = np.arange(-corr_len+1,corr_len,1)

    if show:
        plt.figure()
        for k in range(int(K)):
            plt.subplot(int(K), 1, k+1)
            plt.plot(t_axis, corr_record[k, :])
        plt.show()

    return corr_record


def step_wise_analysis(spk,label,para,smooth=2,no_background=True): # delay * X * Y
    # print(spk.shape)
    sizex = spk.shape[1]
    sizey = spk.shape[2]
    spk = spk.reshape(spk.shape[0], -1)
    spk_tmp = copy.deepcopy(spk)
    alpha=1
    for i in range(smooth - 1):
        alpha *= 0.5
        spk_tmp[i + 1:] += alpha * spk[:-i - 1]
    K = np.max(label) + 1
    estimator = KMeans(n_clusters=int(K), init='k-means++')
    estimator.fit(spk_tmp.T)
    label_pred = estimator.labels_
    label_pred_img = label_pred.reshape(sizex, sizey)
    if no_background == True: # this make plot better!!! note spk & spk_tmp below
        idx = np.where(label.reshape(-1)!=0)[0]
        spk1 = spk[:,idx]
        spk_tmp1 = copy.deepcopy(spk_tmp)[:,idx]
        label_pred = label_pred[idx]

    synchrony_score = silhouette_score(spk1.T, label_pred, metric=victor_purpura_metric, q = 1/3) 
    rate_score = silhouette_score(spk1.T, label_pred, metric=victor_purpura_metric, q=0) 
    km_score = silhouette_score(spk_tmp1.T, label_pred) 
    results = coloring(spk.reshape(-1,sizex,sizey)[None,...], para, back=1,show=False)
    inertia = estimator.inertia_
    return label_pred_img, synchrony_score,rate_score, km_score, inertia, results


def step_wise_analysis_dynamics(spk, label, para,smooth=2, no_background=True): # delay * X * Y
    # print(spk.shape)
    sizex = spk.shape[1]
    sizey = spk.shape[2]
    spk = spk.reshape(spk.shape[0], -1)
    spk_tmp = copy.deepcopy(spk)
    alpha=1
    for i in range(smooth - 1):
        alpha *= 0.5
        spk_tmp[i + 1:] += alpha * spk[:-i - 1]
    K = int(np.max(label) + 1)
    estimator = KMeans(n_clusters=int(K), init='k-means++')
    estimator.fit(spk_tmp.T)
    label_pred = estimator.labels_

    idxes = []
    max_num = 0
    for i in range(K):
        idx = np.where(label_pred == i)[0]
        idxes.append(idx)
        if max_num < len(idx):
            max_num = len(idx)
            back_ground_id = i
    if back_ground_id != 0:
        label_pred[idxes[0]] = back_ground_id
        label_pred[idxes[back_ground_id]] = 0
    label_pred_img = label_pred.reshape(sizex, sizey)
    
    if no_background == True: # this make plot better!!! note spk & spk_tmp below
        idx = np.where(label.reshape(-1)!=0)[0]
        spk1 = spk[:,idx]
        label_pred = label_pred[idx]

    synchrony_score = silhouette_score(spk1.T,label_pred,metric=victor_purpura_metric,q = 1/3) # note
    rate_score = silhouette_score(spk1.T, label_pred, metric=victor_purpura_metric, q=0)  # note
    results = coloring(spk.reshape(-1,sizex,sizey)[None,...],para,back=1,show=False)
    return label_pred_img, synchrony_score,rate_score, results


def draw_step_wise_analysis(grp,syn,r,clr):
    n = len(grp)
    ncol = n//10
    plt.figure()
    for i in range(n):
        plt.subplot(11,2*ncol,2*i+1)
        plt.imshow(grp[i])
        plt.axis('off')
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


def feature_neuron(data='shapes'): # shape: 10000:10000:10000
    if data == 'shapes':
        print('finding feature neurons in shapes net')
        # net = torch.load('./tmp_net/shapes_bae_0.8_net.pty')
        # net = torch.load('./tmp_net/shape_clrnet_net.pty')
        #net = torch.load('../tmp_net/shape_clrnet_spikes_net.pty').to(device)
        net = torch.load('./tmp_net/shapes_svae_gaussian_0.8_1500_net.pty').to(device)
        single = gain_verf_dataset("./tmp_data", "shapes")
        hidden_size = (30, 50)
        sp_sum_record = []
        plt.figure()
        print(torch.tensor(single[0][0]).size())
        for sp in range(3):
            sp_sum = np.zeros(hidden_size)
            print(len(single[sp]))
            for i in range(len(single[sp])):
                x, spike, mem, log_var = net(torch.tensor(single[sp][i], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
                sp_sum += np.array(spike.reshape(hidden_size[0], hidden_size[1]).cpu().detach())
            sp_sum_record.append(sp_sum / len(single[sp]))

        bg = 0
        for sp in range(3):
            plt.subplot(1, 3, sp + 1)
            # plt.imshow(sp_sum_record[sp]-bg, vmin=-0.5, vmax=0.5)
            plt.imshow(sp_sum_record[sp]-bg)
            plt.colorbar(shrink=0.8)
            plt.axis('off')
        plt.show()

        return sp_sum_record


def feature_neuron_with_net(net, hidden_size=(30, 40), vmin=-1, vmax=1, data='shapes'): # shape: 10000:10000:10000
    if data == 'shapes':
        print('finding feature neurons in shapes net')
        single = gain_verf_dataset("../tmp_data", data)
        
        sp_sum_record = []
        plt.figure()
        print(torch.tensor(single[0][0]).size())
        for sp in range(3):
            sp_sum = np.zeros(hidden_size)
            print(len(single[sp]))
            for i in range(len(single[sp])):
                _, spike, _ = net(torch.tensor(single[sp][i], dtype=torch.float32, device=device).reshape(1, -1))
                sp_sum += np.array(spike.reshape(hidden_size[0], hidden_size[1]).cpu().detach())
            sp_sum_record.append(sp_sum / len(single[sp]))

        bg = (1 / 3) * (sp_sum_record[0] + sp_sum_record[1] + sp_sum_record[2])
        for sp in range(3):
            plt.subplot(1, 3, sp + 1)
            # plt.imshow(sp_sum_record[sp], vmin=0, vmax=1.0)
            plt.imshow(sp_sum_record[sp])
            plt.colorbar(shrink=0.8)
            plt.axis('off')
        plt.show()

        plt.figure()
        for sp in range(3):
            plt.subplot(1, 3, sp + 1)
            plt.imshow(sp_sum_record[sp] - bg, vmin=vmin, vmax=vmax)
            plt.colorbar(shrink=0.8)
            plt.axis('off')
        plt.show()

        return sp_sum_record
