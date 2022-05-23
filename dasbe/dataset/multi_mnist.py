import numpy as np
import matplotlib.pyplot as plt
from plot_tools import plot_groups, plot_input_image
import h5py
import os
import os.path
np.random.seed(9825619)

data_dir = "./tmp_data"

# Load the MNIST Dataset as prepared by the brainstorm data script
# You will need to run brainstorm/data/create_mnist.py first
with h5py.File(os.path.join(data_dir, 'MNIST.hdf5'), 'r') as f:
    mnist_digits = f['normalized_full/training/default'][0, :]
    mnist_targets = f['normalized_full/training/targets'][:]
    mnist_digits_test = f['normalized_full/test/default'][0, :]
    mnist_targets_test = f['normalized_full/test/targets'][:]

def crop(d):
    return d[np.sum(d, 1) != 0][:, np.sum(d, 0) != 0]

def generate_multi_mnist_img(digit_nrs, size=(60, 60), test=False, binarize_threshold=0.5):
    if not test:
        digits = [crop(mnist_digits[nr].reshape(28, 28)) for nr in digit_nrs]
    else:
        digits = [crop(mnist_digits_test[nr].reshape(28, 28)) for nr in digit_nrs]
        
    
    flag = False
    while not flag:
        img = np.zeros(size)
        grp = np.zeros(size)
        mask = np.zeros(size)
        k = 1

        for i, digit in enumerate(digits):
            h, w = size
            sy, sx = digit.shape
            x = np.random.randint(0, w-sx+1)
            y = np.random.randint(0, h-sy+1)
            region = (slice(y,y+sy), slice(x,x+sx))
            m = digit >= binarize_threshold
            img[region][m] = 1  
            mask[region][m] += 1  
            grp[region][m] = k      
            k += 1
        if len(digit_nrs) <= 1 or (mask[region][m] > 1).sum() / (mask[region][m] >= 1).sum() < 0.2:
            flag = True
        
    grp[mask > 1] = 0  # ignore overlap regions
    return img, grp

np.random.seed(36520)
nr_digits = 3
mnist_size = 60000 
nr_training_examples = 60000
nr_test_examples = 10000
nr_single_examples = 300000
size = (48, 48)

data = np.zeros((60000,) + size, dtype=np.float32)
grps = np.zeros_like(data)
# targets = np.zeros((60000, nr_digits), dtype=np.int)
for i in range(60000):
    digit_nrs = np.random.randint(0, 60000, nr_digits)
    data[i, :, :], grps[i, :, :] = generate_multi_mnist_img(digit_nrs, size=size)
    # targets[i, :] = mnist_targets[0, digit_nrs, 0]
    
data_test = np.zeros((10000,) + size, dtype=np.float32)
grps_test = np.zeros_like(data_test)
# targets_test = np.zeros((1, 10000, nr_digits), dtype=np.int)
for i in range(10000):
    digit_nrs = np.random.randint(0, 10000, nr_digits)
    data_test[i, :, :], grps_test[i, :, :] = generate_multi_mnist_img(digit_nrs, size=size, test=True)
    # targets_test[0, i, :] = mnist_targets_test[0, digit_nrs, 0]
    
data_single = np.zeros((nr_single_examples,) + size, dtype=np.float32)
grps_single = np.zeros_like(data_single )
# targets_single = np.zeros((1, nr_single_examples, 1), dtype=np.int)
for i in range(nr_single_examples):
    data_single [i, :, :], grps_single[i, :, :] = generate_multi_mnist_img([i % mnist_size], size=size)
    # targets_single[0, i, :] = mnist_targets[0, i, 0]

with h5py.File(os.path.join(data_dir, 'multi_mnist.h5'), 'w') as f:
    single = f.create_group('train_single')
    single.create_dataset('default', data=data_single, compression='gzip', chunks=(100,) + size)
    single.create_dataset('groups', data=grps_single, compression='gzip', chunks=(100,) + size)
    # single.create_dataset('targets', data=targets_single, compression='gzip', chunks=(1, 100, 1))
    
    train = f.create_group('train_multi')
    train.create_dataset('default', data=data, compression='gzip', chunks=(100,) + size)
    train.create_dataset('groups', data=grps, compression='gzip', chunks=(100,) + size)
    # train.create_dataset('targets', data=targets, compression='gzip', chunks=(1, 100, nr_digits))
    
    test = f.create_group('test')
    test.create_dataset('default', data=data_test, compression='gzip', chunks=(100,) + size)
    test.create_dataset('groups', data=grps_test, compression='gzip', chunks=(100,) + size)
    # test.create_dataset('targets', data=targets_test, compression='gzip', chunks=(1, 100, nr_digits))
