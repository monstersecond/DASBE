import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path
import copy


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

directions = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, -1], [-1, 1]]

def generate_shapes_image(width, height, T, nr_shapes=3):
    """
    T: 运动序列的长度
    """
    img = np.zeros((T, height, width))
    grp = np.zeros_like(img)
    k = 1

    # 本次移动选择的形状
    cur_shapes = [shapes[np.random.randint(0, len(shapes))] for _ in range(nr_shapes)]  # 存储本张图片中所有的形状 
    # cur_shapes = [shapes[0], shapes[1], shapes[2]]
    # 本次移动选择的方向
    cur_directions = [copy.deepcopy(directions[np.random.randint(0, len(directions))]) for _ in range(nr_shapes)]  # 每个形状的运动方向
    # cur_directions = [[1, 0], [1, 0], [0, 1]]
    # 本次移动每个形状的位置
    cur_pos = []
    for i in range(nr_shapes):
        sy, sx = cur_shapes[i].shape    # shape大小
        x = np.random.randint(0, width - sx + 1)
        y = np.random.randint(0, height - sy + 1)
        cur_pos.append([x, y])

    for t in range(T):
        k = 1
        for i in range(nr_shapes):
            sy, sx = cur_shapes[i].shape    # shape大小
            x = cur_pos[i][0]        # shape所处位置
            y = cur_pos[i][1]
            region = (slice(y, y+sy), slice(x, x+sx))
            img[t][region][cur_shapes[i] != 0] += 1
            grp[t][region][cur_shapes[i] != 0] = k
            k += 1

            # update cur_pos according to cur_directions
            x += cur_directions[i][0]
            if x > width - sx:
                x = width - sx - 1
                cur_directions[i][0] = -cur_directions[i][0]
            elif x < 0:
                x = 1
                cur_directions[i][0] = -cur_directions[i][0]

            y += cur_directions[i][1]
            if y > height - sy:
                y = height - sy - 1
                cur_directions[i][1] = -cur_directions[i][1]
            elif y < 0:
                y = 1
                cur_directions[i][1] = -cur_directions[i][1]
            cur_pos[i][0] = x
            cur_pos[i][1] = y

        grp[t][img[t] > 1] = 0
        img[t] = img[t] != 0
        
    return img, grp


def generate_shapes_image_by_idx(width, height, T, shape_idx):
    """
    T: 运动序列的长度
    """
    img = np.zeros((T, height, width))
    grp = np.zeros_like(img)
    k = 1
    nr_shapes = 1

    # 本次移动选择的形状
    cur_shapes = [shapes[shape_idx]]  # 存储本张图片中所有的形状 
    # 本次移动选择的方向
    cur_directions = [copy.deepcopy(directions[np.random.randint(0, len(directions))]) for _ in range(nr_shapes)]  # 每个形状的运动方向
    # 本次移动每个形状的位置
    cur_pos = []
    for i in range(nr_shapes):
        sy, sx = cur_shapes[i].shape    # shape大小
        x = np.random.randint(0, width - sx + 1)
        y = np.random.randint(0, height - sy + 1)
        cur_pos.append([x, y])

    for t in range(T):
        k = 1
        for i in range(nr_shapes):
            sy, sx = cur_shapes[i].shape    # shape大小
            x = cur_pos[i][0]        # shape所处位置
            y = cur_pos[i][1]
            region = (slice(y, y+sy), slice(x, x+sx))
            img[t][region][cur_shapes[i] != 0] += 1
            grp[t][region][cur_shapes[i] != 0] = k
            k += 1

            # update cur_pos according to cur_directions
            x += cur_directions[i][0]
            if x > width - sx:
                x = width - sx - 1
                cur_directions[i][0] = -cur_directions[i][0]
            elif x < 0:
                x = 1
                cur_directions[i][0] = -cur_directions[i][0]

            y += cur_directions[i][1]
            if y > height - sy:
                y = height - sy - 1
                cur_directions[i][1] = -cur_directions[i][1]
            elif y < 0:
                y = 1
                cur_directions[i][1] = -cur_directions[i][1]
            cur_pos[i][0] = x
            cur_pos[i][1] = y

        grp[t][img[t] > 1] = 0
        img[t] = img[t] != 0
        
    return img, grp


# plt.figure()
# plt.ion()
# T = 10
# img, grp = generate_shapes_image_by_idx(28, 28, T, 2)
# for t in range(T):
#     plt.cla()
#     plt.imshow(img[t])
#     plt.pause(0.01) 

# plt.ioff()
# plt.show()

np.random.seed(265076)
nr_train_examples = 60
nr_test_examples = 60
nr_single_examples = 60
width = 28
height = 28
nr_shapes = 3
T_max = 1000

data = np.zeros((nr_train_examples, T_max, height, width), dtype=np.float32)
grps = np.zeros_like(data)
for i in range(nr_train_examples):
    data[i, :, :, :], grps[i, :, :, :] = generate_shapes_image(width, height, T_max, nr_shapes)
    
data_test = np.zeros((nr_test_examples, T_max, height, width), dtype=np.float32)
grps_test = np.zeros_like(data_test)
for i in range(nr_test_examples):
    data_test[i, :, :, :], grps_test[i, :, :, :] = generate_shapes_image(width, height, T_max, nr_shapes)

data_single = np.zeros((nr_single_examples, T_max, height, width), dtype=np.float32)
grps_single = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single[i, :, :, :], grps_single[i, :, :, :] = generate_shapes_image(width, height, T_max, 1)

data_single_0 = np.zeros((nr_single_examples, T_max, height, width), dtype=np.float32)
grps_single_0 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_0[i, :, :, :], grps_single_0[i, :, :, :] = generate_shapes_image_by_idx(width, height, T_max, 0)

data_single_1 = np.zeros((nr_single_examples, T_max, height, width), dtype=np.float32)
grps_single_1 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_1[i, :, :, :], grps_single_1[i, :, :, :] = generate_shapes_image_by_idx(width, height, T_max, 1)

data_single_2 = np.zeros((nr_single_examples, T_max, height, width), dtype=np.float32)
grps_single_2 = np.zeros_like(data_single)
for i in range(nr_single_examples):
    data_single_2[i, :, :, :], grps_single_2[i, :, :, :] = generate_shapes_image_by_idx(width, height, T_max, 2)


import h5py

chunk_size = 10
with h5py.File(os.path.join(data_dir, 'moving_shapes.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(chunk_size, T_max, height, width))
    single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(chunk_size, T_max, height, width))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(chunk_size, T_max, height, width))
    train.create_dataset('groups', data=grps, compression='gzip', chunks=(chunk_size, T_max, height, width))
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(chunk_size, T_max, height, width))
    test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(chunk_size, T_max, height, width))

    # 分别生成三个验证数据集，每个数据集只包含一个
    single_0 = f.create_group('train_single_0')
    single_0.create_dataset('default', data=data_single_0, compression='gzip', chunks=(chunk_size, T_max, height, width))
    single_0.create_dataset('groups', data=grps_single_0, compression='gzip', chunks=(chunk_size, T_max, height, width))

    single_1 = f.create_group('train_single_1')
    single_1.create_dataset('default', data=data_single_1, compression='gzip', chunks=(chunk_size, T_max, height, width))
    single_1.create_dataset('groups', data=grps_single_1, compression='gzip', chunks=(chunk_size, T_max, height, width))

    single_2 = f.create_group('train_single_2')
    single_2.create_dataset('default', data=data_single_2, compression='gzip', chunks=(chunk_size, T_max, height, width))
    single_2.create_dataset('groups', data=grps_single_2, compression='gzip', chunks=(chunk_size, T_max, height, width))
