"""
Corner problem:

Reference
[1] David P. Reichert and Thomas Serre, Neuronal Synchrony in Complex-Valued Deep Networks, ICLR 2014
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path

np.random.seed(746519283)

width = 28
height = 28

corner = np.zeros((5, 5))
corner[:2, :] = 1.0
corner[:, :2] = 1.0

corners = [
    corner.copy(),
    corner[::-1, :].copy(),
    corner[:, ::-1].copy(),
    corner[::-1, ::-1].copy()
]

square = np.zeros((20, 20))
square[:5, :5] = corners[0]
square[-5:, :5] = corners[1]
square[:5, -5:] = corners[2]
square[-5:, -5:] = corners[3]

def generate_corners_image(width, height, nr_squares=1, nr_corners=4):
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    k = 1
    
    for i in range(nr_squares):
        x = np.random.randint(0, width-19)
        y = np.random.randint(0, height-19)
        region = (slice(y,y+20), slice(x,x+20))
        img[region][square != 0] += 1
        grp[region][square != 0] = k        
        k += 1
    
    for i in range(nr_corners):
        x = np.random.randint(0, width-4)
        y = np.random.randint(0, height-4)
        corner = corners[np.random.randint(0, 4)]
        region = (slice(y,y+5), slice(x,x+5))
        img[region][corner != 0] += 1
        grp[region][corner != 0] = k
        k += 1
        
    grp[img > 1] = 0
    img = img != 0
    return img, grp

data_dir = "./tmp_data"

nr_train_examples = 60000
nr_test_examples = 10000
nr_single_examples = 15000

width = 28
height = 28
nr_squares = 1
nr_corners = 4

data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
grps = np.zeros_like(data)
for i in range(nr_train_examples):
    data[i, :, :], grps[i, :, :] = generate_corners_image(width, height, nr_squares, nr_corners)
    
data_test = np.zeros((nr_test_examples, height, width), dtype=np.float32)
grps_test = np.zeros_like(data_test)
for i in range(nr_test_examples):
    data_test[i, :, :], grps_test[i, :, :] = generate_corners_image(width, height, nr_squares, 
                                                                                nr_corners)

data_single = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single = np.zeros_like(data_single)
for i in range(nr_single_examples // 7):
    data_single[i, :, :], grps_single[i, :, :] = generate_corners_image(width, height, 1, 0)
for i in range(nr_single_examples // 7, nr_single_examples):
    data_single[i, :, :], grps_single[i, :, :] = generate_corners_image(width, height, 0, 1)

shuffel_idx = np.arange(nr_single_examples)
np.random.shuffle(shuffel_idx)
data_single = data_single[shuffel_idx, :]
grps_single = grps_single[shuffel_idx, :]

with h5py.File(os.path.join(data_dir, 'corners.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(100, height, width))
    single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(100, height, width))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100, height, width))
