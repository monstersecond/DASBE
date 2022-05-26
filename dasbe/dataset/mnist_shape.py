import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
np.random.seed(985619)
import h5py
import os
import os.path

data_dir = "./tmp_data"

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



with h5py.File(os.path.join(data_dir, 'MNIST.hdf5'), 'r') as f:
    mnist_digits = f['normalized_full/training/default'][0, :]
    targets = f['normalized_full/training/targets'][:]
    mnist_digits_test = f['normalized_full/test/default'][0, :]
    targets_test = f['normalized_full/test/targets'][:]


def generate_mnist_shape_img(digit_nr, nr_shapes=1, test=False):
    if digit_nr is None:
        img = np.zeros((28, 28), dtype=np.float)
    elif not test:
        img = (mnist_digits[digit_nr].reshape(28, 28) > 0.5).astype(np.float)
    else:
        img = (mnist_digits_test[digit_nr].reshape(28, 28) > 0.5).astype(np.float)
    
    
    grp = (img > 0.5).astype(np.float)
    mask = grp.copy()
    k = 2
    
    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy, sx = shape.shape
        x = np.random.randint(0, 28-sx+1)
        y = np.random.randint(0, 28-sy+1)
        region = (slice(y,y+sy), slice(x,x+sx))
        img[region][shape != 0] = 1
        mask[region][shape != 0] += 1
        grp[region][shape != 0] = k      
        k += 1
        
    grp[mask > 1] = 0
    return img, grp

np.random.seed(985619)
nr_shapes = 1
nr_training_examples = 60000
nr_test_examples = 10000
nr_single_examples = 20000


data = np.zeros((nr_training_examples, 28, 28), dtype=np.float32)
grps = np.zeros_like(data)
for i in range(nr_training_examples):
    data[i, :, :], grps[i, :, :] = generate_mnist_shape_img(i, nr_shapes)
    
data_test = np.zeros((nr_test_examples, 28, 28), dtype=np.float32)
grps_test = np.zeros_like(data_test)
for i in range(nr_test_examples):
    data_test[i, :, :], grps_test[i, :, :] = generate_mnist_shape_img(i, nr_shapes, test=True)
    

data_single = np.zeros((nr_single_examples, 28, 28), dtype=np.float32)
grps_single = np.zeros_like(data_single)
for i in range(nr_single_examples // 2):
    digit_nr = np.random.randint(0, 60000)
    data_single[i, :, :], grps_single[i, :, :] = generate_mnist_shape_img(digit_nr, 0)
for i in range(nr_single_examples // 2, nr_single_examples):
    data_single[i, :, :], grps_single[i, :, :] = generate_mnist_shape_img(None, 1)

shuffel_idx = np.arange(nr_single_examples)
np.random.shuffle(shuffel_idx)
data_single = data_single[shuffel_idx, :]
grps_single = grps_single[shuffel_idx, :]

with h5py.File(os.path.join(data_dir, 'mnist_shape.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(100, 28, 28))
    single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(100, 28, 28))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100, 28, 28))
    train.create_dataset('groups', data=grps, compression='gzip', chunks=(100, 28, 28))
    
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100, 28, 28))
    test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100, 28, 28))
    

