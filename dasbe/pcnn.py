import numpy as np
from analysis import draw_spikes
import matplotlib.pyplot as plt
import scipy.signal as sgn
from dataset_utils import gain_dataset
from sklearn.cluster import KMeans
from analysis import evaluate_grouping


def salt_pepper_noise(X, p=0.9):
    mask = np.random.rand(X.shape[0],X.shape[1])
    #mask = np.where(mask >= p, 1, 0)
    mask = (mask >= p) # 0.5
    X = mask * X
    #X = np.where(X+mask > 0, 1, 0)
    return X


def gain_weights(height, width, rad=4):
    w = np.zeros((height, width, height, width))
    for i in range(height):
        for j in range(width):
            for k in range(i, height):
                for l in range(width):
                    if (i != k or j != l) and np.abs(i - k) <= rad and np.abs(j - l) <= rad:
                        w[i, j, k, l] = w[k, l, i, j] = np.exp(-np.sqrt((i - k)**2 + (j - l)**2))
    return w


def gain_filter_weights(height, width):
    i = height // 2
    j = width // 2
    w = np.zeros((height, width))
    for k in range(height):
        for l in range(width):
            if i != k or j != l:
                w[k, l] = np.exp(-np.sqrt((i - k)**2 + (j - l)**2))
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

    # (H, W) = (H, W) + (H, W, H, W).sum(3).sum(2)
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

    # y = np.random.randint(0, 2, size=(H, W))
    # F1 = np.random.random(size=(H, W))
    # F2 = np.random.random(size=(H, W))
    # L = np.random.random(size=(H, W))
    # y = F1 = F2 = L = np.zeros((H, W))
    y = np.zeros((H, W))
    F1 = np.zeros((H, W))
    F2 = np.zeros((H, W))
    L = np.zeros((H, W))
    theta = np.ones((H, W)) * theta_

    for _ in range(T):
        # F1 = pcnn_update(y, w_F1, F1, tau_F, V_F, H, W)
        # F2 = salt_pepper_noise(x, 0.1) * w_In
        # F2 = pcnn_update(salt_pepper_noise(x, 0.1), w_F2 * w_In, F2, tau_F, V_F, H, W) * 0.5
        F1 = pcnn_update(y, w_F1, F1, tau_F, V_F, H, W) * 0
        F2 = salt_pepper_noise(x, 0.2)  #pcnn_update(salt_pepper_noise(x, 0.1), w_F2, F2, tau_F, V_F, H, W) * 0.5
        L = pcnn_update(y, w_L, L, tau_L, V_L, H, W)
        U = (F1 + F2) * (1 + beta * L) - min(y.sum(), 1) * w_I #* w_I # + np.random.random(size=(H, W)) * w_random
        assert(U.shape == (H, W))
        theta = np.exp(- 1 / tau_theta) * theta + V_theta * y
        assert (theta.shape == (H, W))
        y = np.where(U - theta >= 0, 1, 0)

        s_record.append(y)
        assert(y.shape == (H, W))
        # theta = np.exp(- 1 / tau_theta) * theta + V_theta * y
        # assert(theta.shape == (H, W))
    return s_record


def pcnn_easy(x, T, beta, w_L, V_L, V_theta, tau_L, tau_theta, tau_I, theta_, w_I, w_random):
    """
    x: (H, W) image
    T: maximum iteration step
    w_L: (H, W, H, W)
    V_L, V_theta: constant
    tau_F, tau_L, tau_theta: constant
    theta_: constant for theta
    w_I: global inhibitory constant
    w_random: random noise weight
    beta: linking coefficient
    """
    H, W = x.shape
    assert(w_L.shape == (H, W, H, W))

    s_record = []
    v_record = []
    y = L = np.zeros((H, W))
    theta = np.ones((H, W)) * theta_
    I = 0

    for _ in range(T):
        L = pcnn_update(y, w_L, L, tau_L, V_L, H, W)
        #U = x * (1 + beta * L) - y.sum() * w_I # all clear
        I = I * np.exp(- 1 / tau_I) + w_I * y.sum()
        # U = x * (1 + beta * (L + np.random.random(size=(H, W)) * w_random)) - I  # all noisy
        U = salt_pepper_noise(x, 0.1) / 4 * (1 + beta * L) - I  # all noisy
        print("1+beta*L: ", (1 + beta * L)[5, 5])
        # U = x * (1 + beta * (L + np.random.random(size=(H, W)) * w_random)) - y.sum() * w_I  # all noisy
        # print(U.max())
        #U = salt_pepper_noise(x,p=0.9) * (1 + beta * (L + np.random.random(size=(H, W)) * w_random)) - y.sum() * w_I
        #U = salt_pepper_noise(x, p=0.5) * (1 + beta * L) - y.sum() * w_I
        assert(U.shape == (H, W))
        theta = np.exp(-1 / tau_theta) * theta + V_theta * y
        assert (theta.shape == (H, W))
        y = np.where(U - theta >= 0, 1, 0)
        s_record.append(y)
        assert(y.shape == (H, W))
        # theta = np.exp(-1 / tau_theta) * theta + V_theta * y
        # assert(theta.shape == (H, W))
        v_record.append(I)
    return s_record, v_record


def pcnn_easier(x, T, beta, V_L, V_theta, tau_L, tau_theta, tau_I, theta_=1, w_I=1, p=0.1):
    """
    x: (H, W) image
    T: maximum iteration step
    V_L, V_theta: constant
    theta_: constant for theta
    w_I: global inhibitory constant
    w_random: random noise weight
    """
    H, W = x.shape
    w = [[0.5,1,0.5],[1,0,1],[0.5,1,0.5]]

    s_record = []
    y = L = np.zeros((H, W))
    theta = np.ones((H, W)) * theta_

    for t in range(T):

        theta = np.exp(-1 / tau_theta) * theta + V_theta * y
        # print(t)
        while True:
            y_post = y
            L = np.exp(- 1 / tau_L) * L + V_L * sgn.convolve(y, w, mode='same')
            # I = I * np.exp(- 1 / tau_I) + w_I * y.sum()
            # noise = np.random.random(size=(H, W))
            U = salt_pepper_noise(x, p) * (1 + beta * L) - min(y.sum(), 1) * w_I  # + w_random * noise  # all noisy
            # U = x * (1 + beta * L) - min(y.sum(), 1) * w_I  # + w_random * noise  # all noisy
            y = np.where(U - theta >= 0, 1, 0)
            # if (y_post == y).all():
            # if np.abs(y_post - y).sum() <= 20:
            if True:
                break
        s_record.append(y)
    return s_record


def pcnn_review(x, T):
    H, W = x.shape
    w = gain_filter_weights(3, 3)

    s_record = []
    
    Y = np.zeros((H, W))
    F = np.zeros((H, W))
    U = np.zeros((H, W))
    L = np.zeros((H, W))
    E = F + 1

    for _ in range(T):
        K = sgn.convolve(Y, w, mode='same')
        assert(K.shape == (H, W))
        F = np.exp(-0.2) * F + 0.1 * K + salt_pepper_noise(x, 0.01)
        assert(F.shape == (H, W))
        L = np.exp(-0.5) * L + 0.2 * K
        assert(L.shape == (H, W))
        U = F * (1 + 0.8 * L) - 0.01 * Y.sum()
        assert(U.shape == (H, W))
        E = np.exp(-0.2) * E + 6. * Y
        assert(E.shape == (H, W))
        Y = np.where(U > E, 1.0, 0.0)
        assert(Y.shape == (H, W))

        s_record.append(Y)

    return s_record


def draw_pcnn(spike):
    lens = len(spike)
    plt.figure()
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(spike[-100+i], vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
    plt.savefig("./pcnn.png")


def not_easy():
    w = gain_weights(28, 28, rad=4)
    assert (w.shape == (28, 28, 28, 28))
    immax = 1
    immin = 0

    img = np.ones((28, 28)) * immin
    img[3:10, 3:10] = immax
    img[3:10, 17:25] = immax
    img[17:25, 3:10] = immax
    img[17:25, 17:25] = immax

    # pcnn(x, T, beta, w_F1, w_F2, w_L, V_F, V_L, V_theta, tau_F, tau_L, tau_theta, theta_, w_I, w_random):
    # spike = pcnn(x=img, T=100, beta=0.2, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=0.2, V_theta=6, tau_F=8, tau_L=2,
    #              tau_theta=5, theta_=1, w_I=1, w_random=0.1)
    spike = pcnn(x=img, T=500, beta=0.2, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=6, tau_F=8, tau_L=2,
                 tau_theta=8, theta_=2, w_I=0.01, w_random=0.1, w_In=1/4)
    draw_pcnn(spike)


def easy():
    w = gain_weights(28, 28, rad=4)
    assert (w.shape == (28, 28, 28, 28))

    immax = 1
    immin = 0.1

    img = np.ones((28, 28)) * immin
    img[3:10, 3:10] = immax
    img[3:10, 16:25] = immax
    img[16:25, 3:10] = immax
    img[16:25, 16:25] = immax

    #            x, T, beta, w_F1, w_F2, w_L, V_F, V_L, V_theta, tau_F, tau_L, tau_theta, theta_, w_I, w_random
    # spike = pcnn(img, 100, 0.2, w, w, w, 1, 1, 100, 5, 5, 5, 1, 1.5, 0.1)

    #            x, T, beta, w_L, V_L, V_theta, tau_L, tau_theta, theta_, w_I, w_random
    spike, v_record = pcnn_easy(img, T=200, beta=0.2, w_L=w, V_L=0.2, V_theta=6, tau_L=2, tau_theta=5, tau_I=2,
                                theta_=1, w_I=1, w_random=0.1)
    # plt.imshow(img)
    draw_pcnn(spike)

    plt.figure()
    plt.plot(v_record)
    plt.show()


def easier():

    immax = 1
    immin = 0

    img = np.ones((28, 28)) * immin
    img[3:10, 3:10] = immax
    img[3:10, 16:25] = immax
    img[16:25, 3:10] = immax
    img[16:25, 16:25] = immax

    #                              x, T, V_L, V_theta, tau_L, tau_theta, tau_I, theta_ = 1, w_I = 1, w_random = 1
    spike = pcnn_easier(img, T=100, beta = 1.2, V_L=1, V_theta=10, tau_L=2, tau_theta=3, tau_I=2, p=0.1,w_I=1.2)
    # plt.imshow(img)
    draw_pcnn(spike)



def shape_not_easy():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "shapes")
    print(multi[0].shape, multi_label[0])
    # print(multi_bars.dtype)
    i = np.random.randint(0, 6000)
    img = multi[i]
    print(img.shape)
    # plt.imshow(img)

    w = gain_weights(28, 28, rad=4)
    assert (w.shape == (28, 28, 28, 28))
    # immax = 1
    # immin = 0

    # img = np.ones((28, 28)) * immin
    # img[3:10, 3:10] = immax
    # img[3:10, 17:25] = immax
    # img[17:25, 3:10] = immax
    # img[17:25, 17:25] = immax

    # pcnn(x, T, beta, w_F1, w_F2, w_L, V_F, V_L, V_theta, tau_F, tau_L, tau_theta, theta_, w_I, w_random):
    # spike = pcnn(x=img, T=100, beta=0.2, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=0.2, V_theta=6, tau_F=8, tau_L=2,
    #              tau_theta=5, theta_=1, w_I=1, w_random=0.1)
    spike = pcnn(x=img, T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=6, tau_F=8, tau_L=2,
                 tau_theta=8, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)
    draw_pcnn(spike)


def bar_not_easy():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "bars")
    print(multi[0].shape, multi_label[0])
    # print(multi_bars.dtype)
    i = np.random.randint(0, 6000)
    img = multi[i]
    print(img.shape)
    # plt.imshow(img)

    w = gain_weights(20, 20, rad=4)
    assert (w.shape == (20, 20, 20, 20))
    # immax = 1
    # immin = 0

    # img = np.ones((28, 28)) * immin
    # img[3:10, 3:10] = immax
    # img[3:10, 17:25] = immax
    # img[17:25, 3:10] = immax
    # img[17:25, 17:25] = immax

    # pcnn(x, T, beta, w_F1, w_F2, w_L, V_F, V_L, V_theta, tau_F, tau_L, tau_theta, theta_, w_I, w_random):
    # spike = pcnn(x=img, T=100, beta=0.2, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=0.2, V_theta=6, tau_F=8, tau_L=2,
    #              tau_theta=5, theta_=1, w_I=1, w_random=0.1)
    spike = pcnn(x=img, T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=10, tau_F=8, tau_L=2,
                 tau_theta=6, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)
    draw_pcnn(spike)


def run_pcnn_review():
    
    immax = 1
    immin = 0

    img = np.ones((28, 28)) * immin
    img[3:10, 3:10] = immax
    img[3:10, 17:25] = immax
    img[17:25, 3:10] = immax
    img[17:25, 17:25] = immax

    spike = pcnn_review(img, T=100)
    # plt.imshow(img)
    draw_pcnn(spike)

    # plt.figure()
    # plt.plot(v_record)
    # plt.show()

def pcnn_AMI_test_shape():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "shapes")
    print(multi[0].shape, multi_label[0])
    w = gain_weights(28, 28, rad=4)
    AMI_score = []
    idx_dict = {}
    for i in range(6000):
        spk = pcnn(x=np.array(multi[i], dtype=np.float32), T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=6, tau_F=8, tau_L=2,
                     tau_theta=8, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)

        pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=100, smooth=2)

        ami_score = evaluate_grouping(multi_label[i], pred_low)
        idx_dict[ami_score] = i
        AMI_score.append(ami_score)
    AMI_score.sort()
    print("shape_pcnn_mean_AMI_score: ", np.array(AMI_score).mean())
    fig = plt.figure()
    plt.plot(np.arange(len(AMI_score)), np.array(AMI_score))
    fig.savefig('./tmp_img/AMI_score_' + 'shape_pcnn' + str(
        np.around(np.array(AMI_score).mean(), decimals=2)) + '.png')

    draw_AMI_selected_images(multi, AMI_score, idx_dict,'shape')
    print('pcnn')

def pcnn_AMI_test_bar():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "bars")
    print(multi[0].shape, multi_label[0])
    w = gain_weights(20, 20, rad=4)
    AMI_score = []
    idx_dict = {}
    for i in range(6000):
        spk = pcnn(x=np.array(multi[i], dtype=np.float32), T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=10, tau_F=8, tau_L=2,
                     tau_theta=6, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)

        pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=100, smooth=2)

        ami_score = evaluate_grouping(multi_label[i], pred_low)
        idx_dict[ami_score] = i
        AMI_score.append(ami_score)
    AMI_score.sort()
    print("bar_pcnn_mean_AMI_score: ", np.array(AMI_score).mean())
    fig = plt.figure()
    plt.plot(np.arange(len(AMI_score)), np.array(AMI_score))
    fig.savefig('./tmp_img/AMI_score_' + '_bar_pcnn_' + str(
        np.around(np.array(AMI_score).mean(), decimals=2)) + '.png')

    draw_AMI_selected_images(multi, AMI_score, idx_dict,'bars')
    print('pcnn')

def pcnn_AMI_test_corner():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "corners")
    print(multi[0].shape, multi_label[0])
    w = gain_weights(28, 28, rad=4)
    AMI_score = []
    idx_dict = {}
    for i in range(6000):
        spk = pcnn(x=np.array(multi[i], dtype=np.float32), T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=10, tau_F=8, tau_L=2,
                     tau_theta=6, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)

        pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=100, smooth=2)

        ami_score = evaluate_grouping(multi_label[i], pred_low)
        idx_dict[ami_score] = i
        AMI_score.append(ami_score)
    AMI_score.sort()
    print("corner_pcnn_mean_AMI_score: ", np.array(AMI_score).mean())
    fig = plt.figure()
    plt.plot(np.arange(len(AMI_score)), np.array(AMI_score))
    fig.savefig('./tmp_img/AMI_score_' + '_corner_pcnn_' + str(
        np.around(np.array(AMI_score).mean(), decimals=2)) + '.png')

    draw_AMI_selected_images(multi, AMI_score, idx_dict,'corners')
    print('pcnn')

def pcnn_AMI_test_multimnist():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "multi_mnist")
    print(multi[0].shape, multi_label[0])
    w = gain_weights(48, 48, rad=4)
    AMI_score = []
    idx_dict = {}
    for i in range(6000):
        spk = pcnn(x=np.array(multi[i], dtype=np.float32), T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=10, tau_F=8, tau_L=2,
                     tau_theta=6, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)

        pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=100, smooth=2)

        ami_score = evaluate_grouping(multi_label[i], pred_low)
        idx_dict[ami_score] = i
        AMI_score.append(ami_score)
    AMI_score.sort()
    print("multimnist_pcnn_mean_AMI_score: ", np.array(AMI_score).mean())
    fig = plt.figure()
    plt.plot(np.arange(len(AMI_score)), np.array(AMI_score))
    fig.savefig('./tmp_img/AMI_score_' + '_multimnist_pcnn_' + str(
        np.around(np.array(AMI_score).mean(), decimals=2)) + '.png')

    draw_AMI_selected_images(multi, AMI_score, idx_dict,'multimnist')
    print('pcnn')

def pcnn_AMI_test_mnistshape():
    _, multi, _, _, multi_label, _ = gain_dataset("./tmp_data", "mnist_shape")
    print(multi[0].shape, multi_label[0])
    w = gain_weights(28, 28, rad=4)
    AMI_score = []
    idx_dict = {}
    for i in range(6000):
        spk = pcnn(x=np.array(multi[i], dtype=np.float32), T=500, beta=3, w_F1=w, w_F2=w, w_L=w, V_F=0.2, V_L=1, V_theta=10, tau_F=8, tau_L=2,
                     tau_theta=6, theta_=2, w_I=3.5, w_random=0.1, w_In=1 / 4)

        pred_low = k_means(np.array(spk).astype(np.float64), multi_label[i], show=False, back=100, smooth=2)

        ami_score = evaluate_grouping(multi_label[i], pred_low)
        idx_dict[ami_score] = i
        AMI_score.append(ami_score)
    AMI_score.sort()
    print("mnist_shape_pcnn_mean_AMI_score: ", np.array(AMI_score).mean())
    fig = plt.figure()
    plt.plot(np.arange(len(AMI_score)), np.array(AMI_score))
    fig.savefig('./tmp_img/AMI_score_' + '_mnist_shape_pcnn_' + str(
        np.around(np.array(AMI_score).mean(), decimals=2)) + '.png')

    draw_AMI_selected_images(multi, AMI_score, idx_dict,'mnist_shape')
    print('pcnn')

def k_means(spk,label,show = True, back = 100, smooth=0): #both low & high level
    spk = spk[-back:, :, :]
    print("k_means:  ","X_shape1: ",spk.shape)
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

def draw_AMI_selected_images(multi,AMI_score,idx_dict,name):
    selected_ami = [AMI_score[1], AMI_score[1000], AMI_score[2000], AMI_score[3000], AMI_score[4000], AMI_score[5000],
                    AMI_score[5999]]
    fig = plt.figure()
    for i in range(len(selected_ami)):
        plt.subplot(4, 2, i+1)
        plt.imshow(np.array(multi[idx_dict[selected_ami[i]]], dtype=np.float32))
    fig.savefig('./tmp_img/img_' + 'pcnn' + name + '.png')

if __name__ == "__main__":
    # not_easy()
    # easy()
    #easier()
    #shape_not_easy()
    #run_pcnn_review()
    #pcnn_AMI_test_shape()
    #pcnn_AMI_test_bar()
    pcnn_AMI_test_corner()
    pcnn_AMI_test_mnistshape()
    pcnn_AMI_test_multimnist()
    #bar_not_easy()



