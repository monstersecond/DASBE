"""
Adapted bars problem:

Binary images with a fixed number of randomly placed horizontal and
vertical bars. 
width = height = 20  # 图片大小
nr_norizontal_bars = nr_vertical_bars = 6  # 图片中水平和竖直bar的数量
====================================================================
This python file should run in the diectory where the file exists.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import copy


def generate_bars(width, height, nr_horizontal_bars, nr_vertical_bars):
    """
    generate test bars
    input:
        width: image width
        height: image height
        nr_horizontal_bars: numbers of horizontal bars
        nv_vertical_bars: numbers of vertical bars
    output:
        img: result image
        grp: mask matrix with 1 been selected; 0, not selected
    """
    img = np.zeros((height, width), dtype=np.float)
    grp = np.zeros_like(img)
    
    idx_vert = np.random.choice(np.arange(width), replace=False, size=nr_vertical_bars)
    img[:, idx_vert] = 1.
    k = 1
    for i in idx_vert:
        grp[:, i] = k
        k += 1
    
    idx_horiz = np.random.choice(np.arange(height), replace=False, size=nr_horizontal_bars)
    img[idx_horiz, :] += 1.
    for i in idx_horiz:
        grp[i, :] = k
        k += 1
    
    grp[img > 1] = 0  # 交叉处颜色为0
    img = img != 0
    
    return img, grp

def generate_bars_with_T(width, height, nr_horizontal_bars, nr_vertical_bars,
                        T, v):
    """
    generate test bars with time window size T.
    Randomly choose a bar to move, speed is equal to v.
    input:
        width: image width
        height: image height
        nr_horizontal_bars: numbers of horizontal bars
        nv_vertical_bars: numbers of vertical bars
    output:
        img: result image
        grp: mask matrix with 1 been selected; 0, not selected
    """
    imgs = np.zeros((T, height, width), dtype=np.float)  # 保存长度为T的数据集
    grps = np.zeros_like(imgs) 

    if (np.random.rand() >= 0.5 and nr_vertical_bars > 0) or nr_horizontal_bars == 0:
        hori = False
        nr_vertical_bars -= 1
        nr_vertical_bars = max(nr_vertical_bars, 0)
    else:
        hori = True
        nr_horizontal_bars -= 1
        nr_horizontal_bars = max(nr_horizontal_bars, 0)
    
    img = np.zeros((height, width), dtype=np.float)
    grp = np.zeros_like(img)
    
    idx_vert = np.random.choice(np.arange(width), replace=False, size=nr_vertical_bars)
    img[:, idx_vert] = 1.
    k = 1
    for i in idx_vert:
        grp[:, i] = k
        k += 1
    
    idx_horiz = np.random.choice(np.arange(height), replace=False, size=nr_horizontal_bars)
    img[idx_horiz, :] += 1.
    for i in idx_horiz:
        grp[i, :] = k
        k += 1

    cur_idx = np.random.randint(low=0, high=width)  # 其实的坐标未知

    for t in range(T):
        cur_grp = copy.deepcopy(grp)
        cur_img = copy.deepcopy(img)
        if hori:  # 横向
            cur_grp[cur_idx, :] = k 
            cur_img[cur_idx, :] += 1. 
        else:
            cur_grp[:, cur_idx] = k
            cur_img[:, cur_idx] += 1.

        cur_idx = (cur_idx + v) % width
        
        cur_grp[cur_img > 1] = 0  # 交叉处颜色为0
        cur_img = cur_img != 0

        imgs[t, :] = cur_img
        grps[t, :] = cur_grp

    return imgs, grps

if __name__ == "__main__":
    # plt.savefig('bars_example.png')

    # 设置生成相关参数
    np.random.seed(471958)
    nr_train_examples = 60000
    nr_test_examples = 10000
    nr_single_examples = 200
    width = 20
    height = 20
    nr_vert = 6
    nr_horiz= 6

    # 生成训练集
    data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
    grps = np.zeros_like(data)
    for i in range(nr_train_examples):
        data[i, :, :], grps[i, :, :] = generate_bars(width, height, nr_horiz, nr_vert)

    # 生成测试集
    test_data = np.zeros((nr_test_examples, height, width), dtype=np.float32)
    test_grps = np.zeros_like(test_data)
    for i in range(nr_test_examples):
        test_data[i, :, :], test_grps[i, :, :] = generate_bars(width, height, nr_horiz, nr_vert)

    # 生成只有一条的数据（一半竖直，一半水平）
    single_data = np.zeros((nr_single_examples, height, width), dtype=np.float32)
    single_grps = np.zeros_like(single_data)
    for i in range(nr_single_examples // 2):
        single_data[i, :, :], single_grps[i, :, :] = generate_bars(width, height, 1, 0)
    for i in range(nr_single_examples // 2, nr_single_examples):
        single_data[i, :, :], single_grps[i, :, :] = generate_bars(width, height, 0, 1)
    # 保存只含有横的和只含有纵的分别的数据集
    single_data_hori = copy.deepcopy(single_data[:nr_single_examples//2])
    single_grps_hori = copy.deepcopy(single_grps[:nr_single_examples//2])
    single_data_vert = copy.deepcopy(single_data[nr_single_examples//2:])
    single_grps_vert = copy.deepcopy(single_grps[nr_single_examples//2:])
    # 将两组的数据打乱
    shuffel_idx = np.arange(nr_single_examples)
    np.random.shuffle(shuffel_idx)
    print(shuffel_idx)
    single_data = single_data[shuffel_idx, :]
    single_grps = single_grps[shuffel_idx, :]

    data_dir = "./tmp_data/"
    with h5py.File(os.path.join(data_dir, 'bars.h5'), 'w') as f:
        single = f.create_group('train_single')
        single.create_dataset('default', data=single_data, compression='gzip', chunks=(100, height, width))
        single.create_dataset('groups', data=single_grps, compression='gzip', chunks=(100, height, width))
        train = f.create_group('train_multi')
        train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
        train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
        test = f.create_group('test')
        test.create_dataset('default', data=test_data, compression='gzip', chunks=(100, height, width))
        test.create_dataset('groups', data=test_grps, compression='gzip', chunks=(100, height, width))

        single_hori = f.create_group('train_single_hori')
        single_hori.create_dataset('default', data=single_data_hori, compression='gzip', chunks=(100, height, width))
        single_hori.create_dataset('groups', data=single_grps_hori, compression='gzip', chunks=(100, height, width))
        single_vert = f.create_group('train_single_vert')
        single_vert.create_dataset('default', data=single_data_vert, compression='gzip', chunks=(100, height, width))
        single_vert.create_dataset('groups', data=single_grps_vert, compression='gzip', chunks=(100, height, width))
        