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
from sklearn.cluster import KMeans
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

def salt_pepper_noise(X, p=0.9):
    mask = torch.rand(X.shape[0],X.shape[1], dtype=torch.float32)
    mask = (mask >= p) # 0.5
    X = mask * X
    return X

def gain_weights(height, width, rad=4):
    w = torch.zeros((height, width, height, width),device=device)
    for i in range(height):
        for j in range(width):
            for k in range(i, height):
                for l in range(width):
                    if (i != k or j != l) and np.abs(i - k) <= rad and np.abs(j - l) <= rad:
                        w[i, j, k, l] = w[k, l, i, j] = np.exp(-np.sqrt((i - k)**2 + (j - l)**2))
    return w

def pcnn_update(y, w_F, pre_F, tau_F, V_F, H, W):
    """
    y: (H, W) spikes output of each pixel
        or x: (H, W) input image
    w_F: (H, W, H, W) recurrent synapse weight from spike output
        or u_F: (H, W, H, W) synapse weight from input
        or w_L: (H, W, H, W)
    pre_F: pre_F1: (H, W) previous w_F * y
        or pre_F2: previous u_F * x
        or pre_L: (H, W)
    tau_F: time constant for each pixel
        or tau_L: time constant
    V_F: time constant for I(V, tau, t)
        or V_L: time constant
    H: height
    W: width
    """
    assert(y.shape == (H, W))
    assert(w_F.shape == (H, W, H, W))
    assert(pre_F.shape == (H, W))

    F = (pre_F * np.exp(-1/tau_F)) + ((w_F * y) * V_F).sum(axis=3).sum(axis=2)
    assert(F.shape == (H, W))

    return F

def pcnn(x, T, beta, w_F1, w_F2, w_L, V_F, V_L, V_theta, tau_F, tau_L, tau_theta, theta_, w_I, w_random, w_In):
    """
    x: (H, W) image
    T: maximum iteration step
    w_F1, W_F2, w_L: (H, W, H, W)
    V_F, V_L, V_theta: constant
    tau_F, tau_L, tau_theta: constant
    theta_: constant for theta
    w_I: global inhibitory constant
    w_random: random noise weight
    beta: linking coefficient
    """
    H, W = x.shape
    assert(w_F1.shape == (H, W, H, W))
    assert(w_F2.shape == (H, W, H, W))
    assert(w_L.shape == (H, W, H, W))

    s_record = []

    y = torch.zeros((H, W),device=device)
    F1 = torch.zeros((H, W),device=device)
    L = torch.zeros((H, W),device=device)
    theta = torch.ones((H, W),device=device) * theta_

    for _ in range(T):
        F1 = pcnn_update(y, w_F1, F1, tau_F, V_F, H, W) * 0
        F2 = salt_pepper_noise(x, 0.2)  #pcnn_update(salt_pepper_noise(x, 0.1), w_F2, F2, tau_F, V_F, H, W) * 0.5
        F2 = F2.to(device)
        L = pcnn_update(y, w_L, L, tau_L, V_L, H, W)
        U = (F1 + F2) * (1 + beta * L) - min(y.sum(), 1) * w_I #* w_I # + np.random.random(size=(H, W)) * w_random
        assert(U.shape == (H, W))
        theta = torch.exp(- 1 / torch.from_numpy(np.array(tau_theta))) * theta + V_theta * y
        assert (theta.shape == (H, W))
        y = torch.where(U - theta >= 0, 1, 0)

        s_record.append(np.array(y.cpu()))
        assert(y.shape == (H, W))
    return np.array(s_record)

def k_means(spk,label,show = True, back = 100, smooth=0): #both low & high level
    spk = spk[-back:, :, :]
    # print("k_means:  ","X_shape1: ",spk.shape)
    # spk_tmp = spk
    sizex = spk.shape[1]
    sizey = spk.shape[2]
    spk_tmp = spk
    alpha=1
    for i in range(smooth-1):
        alpha*=0.5
        spk_tmp[i+1:] += alpha * spk[:-i-1]

    spk = spk_tmp
    K = np.max(label)+1
    spk = spk.reshape(spk.shape[0], -1)
    spk = spk.T # (n_samples, n_featrues)
    estimator = KMeans(n_clusters=int(K),init = 'k-means++')

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

def pcnn_on_each_dataset(data_name,para,back=100,smooth=2):
    _, multi, _, _, multi_label, _ = gain_dataset("../../tmp_data", data_name)
    print(multi[0].shape, multi_label[0])

    AMI_mean = []
    AMI_score_5 = []
    # seed_list = np.random.randint(1, 1000, 5)
    seed_list = [111,222,333,444,555]
    for seed in seed_list:  # different random seed
        setup_seed_pytorch(seed)
        w = gain_weights(para['weight_size'], para['weight_size'], rad=4)
        AMI_score = []
        idx_dict = {}
        for i in range(6000):
            start_time1 = datetime.datetime.now()
            spk = pcnn(multi[i], T=para['T'], beta=para['beta'], w_F1=w, w_F2=w, w_L=w, V_F=para['V_F'], V_L=para['V_L'], V_theta=para['V_theta'], tau_F=para['tau_F'], tau_L=para['tau_L'],
                         tau_theta=para['tau_theta'], theta_=para['theta_'], w_I=para['w_I'], w_random=para['w_random'], w_In=para['w_In'])

            pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=back, smooth=smooth)

            ami_score = evaluate_grouping(multi_label[i], pred_low)
            end_time1 = datetime.datetime.now()
            print("time used:", end_time1 - start_time1,data_name, " seed:", seed, ' id:', i, " AMI score:", ami_score)
            idx_dict[ami_score] = i
            AMI_score.append(ami_score)
        AMI_score.sort()
        AMI_mean.append(np.array(AMI_score).mean())
        AMI_score_5.append(AMI_score)

    return np.array(AMI_score_5), np.array(AMI_mean)

data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
para_list = {
    'bars': {'T':500,'beta':3,'V_F':0.2, 'V_L':1, 'V_theta':10, 'tau_F':8, 'tau_L':2,
                 'tau_theta':6, 'theta_':2, 'w_I':3.5, 'w_random':0.1, 'w_In':1 / 4,'weight_size':20},
    'shapes': {'T':500,'beta':3,'V_F':0.2, 'V_L':1, 'V_theta':6, 'tau_F':8, 'tau_L':2,
                 'tau_theta':8, 'theta_':2, 'w_I':3.5, 'w_random':0.1, 'w_In':1 / 4, 'weight_size':28},
    'corners': {'T':500,'beta':3,'V_F':0.2, 'V_L':1, 'V_theta':10, 'tau_F':8, 'tau_L':2,
                 'tau_theta':6, 'theta_':2, 'w_I':3.5, 'w_random':0.1, 'w_In':1 / 4, 'weight_size':28},
    'multi_mnist': {'T':500,'beta':3,'V_F':0.2, 'V_L':1, 'V_theta':10, 'tau_F':8, 'tau_L':2,
                 'tau_theta':6, 'theta_':2, 'w_I':3.5, 'w_random':0.1, 'w_In':1 / 4, 'weight_size':48},
    'mnist_shape': {'T':500,'beta':3,'V_F':0.2, 'V_L':1, 'V_theta':10, 'tau_F':8, 'tau_L':2,
                 'tau_theta':6, 'theta_':2, 'w_I':3.5, 'w_random':0.1, 'w_In':1 / 4, 'weight_size':28}
}

data_name_list=['bars','corners','shapes','multi_mnist','mnist_shape']
start_time = datetime.datetime.now()
for i in range(5):
    print("i = ", i)
    start_time1 = datetime.datetime.now()
    AMI_score, AMI_mean = pcnn_on_each_dataset(data_name_list[i],para_list[data_name_list[i]], back=100)
    np.save('./AMI_npys/ami_pcnn_score_5array_555_'+data_name_list[i],AMI_score)
    np.save('./AMI_npys/ami_pcnn_score_5mean__555_' + data_name_list[i], AMI_mean)
    end_time1 = datetime.datetime.now()
    print("time used:", end_time1-start_time1, data_name_list[i] + ' :  ', AMI_score.shape, 'back=', 100)
end_time = datetime.datetime.now()
print("time used:", end_time-start_time)



