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


def generate_shapes_image(width, height, nr_shapes=3):
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    k = 1
    
    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy, sx = shape.shape
        x = np.random.randint(0, width-sx+1)
        y = np.random.randint(0, height-sy+1)
        region = (slice(y,y+sy), slice(x,x+sx))
        img[region][shape != 0] += 1
        grp[region][shape != 0] = k        
        k += 1
        
    grp[img > 1] = 0
    img = img != 0
    return img, grp


def generate_shapes_image_by_idx(width, height, shape_idx):
    """
    shape_idx指定了是什么形状, 范围在0~2内
    """
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
    return img, grp


np.random.seed(265076)
nr_train_examples = 60000
nr_test_examples = 10000
nr_single_examples = 20000
width = 28
height = 28
nr_shapes = 3

data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
grps = np.zeros_like(data)
for i in range(nr_train_examples):
    data[i, :, :], grps[i, :, :] = generate_shapes_image(width, height, nr_shapes)
    
data_test = np.zeros((nr_test_examples, height, width), dtype=np.float32)
grps_test = np.zeros_like(data_test)
for i in range(nr_test_examples):
    data_test[i, :, :], grps_test[i, :, :] = generate_shapes_image(width, height, nr_shapes)

data_single = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single[i, :, :], grps_single[i, :, :] = generate_shapes_image(width, height, 1)

data_single_0 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_0 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_0[i, :, :], grps_single_0[i, :, :] = generate_shapes_image_by_idx(width, height, 0)

data_single_1 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_1 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_1[i, :, :], grps_single_1[i, :, :] = generate_shapes_image_by_idx(width, height, 1)

data_single_2 = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single_2 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_2[i, :, :], grps_single_2[i, :, :] = generate_shapes_image_by_idx(width, height, 2)


import h5py

with h5py.File(os.path.join(data_dir, 'shapes.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(100, height, width))
    single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(100, height, width))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100, height, width))
    train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, height, width))
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100, height, width))
    test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100, height, width))

    # 分别生成三个验证数据集，每个数据集只包含一个
    single_0 = f.create_group('train_single_0')
    single_0.create_dataset('default', data=data_single_0, compression='gzip', chunks=(100, height, width))
    single_0.create_dataset('groups', data=grps_single_0, compression='gzip', chunks=(100, height, width))

    single_1 = f.create_group('train_single_1')
    single_1.create_dataset('default', data=data_single_1, compression='gzip', chunks=(100, height, width))
    single_1.create_dataset('groups', data=grps_single_1, compression='gzip', chunks=(100, height, width))

    single_2 = f.create_group('train_single_2')
    single_2.create_dataset('default', data=data_single_2, compression='gzip', chunks=(100, height, width))
    single_2.create_dataset('groups', data=grps_single_2, compression='gzip', chunks=(100, height, width))

    
