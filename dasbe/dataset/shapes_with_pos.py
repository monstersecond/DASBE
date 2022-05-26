import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path


np.random.seed(104174)
data_dir = "./tmp_data/"

square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

shapes = [square, triangle, triangle[::-1, :].copy()]

def generate_shapes_image_by_idx(width, height, shape_idx):
    assert(shape_idx >= 0 and shape_idx <= 2)
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    k = 1

    shape = shapes[shape_idx]
    sy, sx = shape.shape
    x = np.random.randint(0, width-sx+1)
    y = np.random.randint(0, height-sy+1)
    region = (slice(y,y+sy), slice(x,x+sx))
    img[region][shape != 0] += 1
    grp[region][shape != 0] = k        
    k += 1
        
    grp[img > 1] = 0
    img = img != 0
    return img, grp, y, x


np.random.seed(265076)
nr_train_examples = 60000
nr_test_examples = 10000
nr_single_examples = 60000
width = 28
height = 28
nr_shapes = 3

data_single_0 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_0 = np.zeros_like(data_single_0)
pos_0 = np.zeros((nr_single_examples, 2))
for i in range(nr_single_examples):
    data_single_0[i, :, :], grps_single_0[i, :, :], pos_0[i, 0], pos_0[i, 1]  = generate_shapes_image_by_idx(width, height, 0)

data_single_1 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_1 = np.zeros_like(data_single_1)
pos_1 = np.zeros((nr_single_examples, 2))
for i in range(nr_single_examples):
    data_single_1[i, :, :], grps_single_1[i, :, :], pos_1[i, 0], pos_1[i, 1] = generate_shapes_image_by_idx(width, height, 1)

data_single_2 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_2 = np.zeros_like(data_single_2)
pos_2 = np.zeros((nr_single_examples, 2))
for i in range(nr_single_examples):
    data_single_2[i, :, :], grps_single_2[i, :, :], pos_2[i, 0], pos_2[i, 1] = generate_shapes_image_by_idx(width, height, 2)


import h5py

with h5py.File(os.path.join(data_dir, 'shapes_with_pos.h5'), 'w') as f:
    
    single_0 = f.create_group('train_single_0')
    single_0.create_dataset('default', data=data_single_0, compression='gzip', chunks=(100, height, width))
    single_0.create_dataset('groups', data=grps_single_0, compression='gzip', chunks=(100, height, width))
    single_0.create_dataset('pos', data=pos_0, compression='gzip', chunks=(100, 2))

    single_1 = f.create_group('train_single_1')
    single_1.create_dataset('default', data=data_single_1, compression='gzip', chunks=(100, height, width))
    single_1.create_dataset('groups', data=grps_single_1, compression='gzip', chunks=(100, height, width))
    single_1.create_dataset('pos', data=pos_1, compression='gzip', chunks=(100, 2))

    single_2 = f.create_group('train_single_2')
    single_2.create_dataset('default', data=data_single_2, compression='gzip', chunks=(100, height, width))
    single_2.create_dataset('groups', data=grps_single_2, compression='gzip', chunks=(100, height, width))
    single_2.create_dataset('pos', data=pos_2, compression='gzip', chunks=(100, 2))

    
