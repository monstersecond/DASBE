"""
This file used to deal with dataset.
"""

import h5py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg


class BindingDataset(Dataset):
    def __init__(self, data_dir, name, is_single=True, train=True, ver=False, is_ver=False
                    , train_ver_splie_rate=0.9, transforms=None, target_transforms=None):
        """
        Init torch dataset.
        data_dir: directory to the dataset h5 files.
        name: name of the data, eg: bars.
        is_single: whether the image contains only one object. (Default True)
        train: load train dataset or val. (Default True)

        split_train_ver: whether split train verification test. (Default False)
        is_val: this is verification test. (default False)
        train_ver_splie_rate: split rate of train and verification test. (Default 0.9)
        """

        super(BindingDataset, self).__init__()

        self.transforms = transforms
        self.target_transforms = target_transforms

        if name == "clevr":
            train_single_data, multi_clevr_data = gain_dataset("", "clevr")
            if is_single:
                self.data = train_single_data
                self.label = train_single_data
            else:
                self.data = multi_clevr_data
                self.label = multi_clevr_data

            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.label = torch.tensor(self.label, dtype=torch.float32)
        else:
            train_single_data, train_multi_data, test_data, train_single_label, train_multi_label, test_label = gain_dataset(data_dir, name)
            if train:
                if is_single:
                    self.data = train_single_data
                    self.label = train_single_label
                else:
                    self.data = train_multi_data
                    self.label = train_multi_label
            else:
                self.data = test_data
                self.label = test_label

            # split train and verification dataset
            if train and ver:
                data_size = int(self.data.shape[0] * train_ver_splie_rate)
                if is_ver:
                    self.data = self.data[data_size:, :]
                    self.label = self.label[data_size:, :]
                else:
                    self.data = self.data[:data_size, :] 
                    self.label = self.label[:data_size, :]
        
            self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(dim=1)
            self.label = torch.tensor(self.label, dtype=torch.float32).unsqueeze(dim=1)

        print(self.data.shape)
        print(self.label.shape)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.label[idx]
        
        if self.transforms:
            img = self.transforms(img)
        
        if self.target_transforms:
            label = self.target_transforms(label)
        
        return img, label 

    def __len__(self):
        return self.data.shape[0]


class ClrNetDataset(Dataset):
    def __init__(self, data_dir, name, is_single=True, train=True, ver=False, is_ver=False
                    , train_ver_splie_rate=0.9, transforms=None, target_transforms=None):
        """
        Init torch dataset.
        data_dir: directory to the dataset h5 files.
        name: name of the data, eg: bars.
        is_single: whether the image contains only one object. (Default True)
        train: load train dataset or val. (Default True)

        split_train_ver: whether split train verification test. (Default False)
        is_val: this is verification test. (default False)
        train_ver_splie_rate: split rate of train and verification test. (Default 0.9)
        """

        super(ClrNetDataset, self).__init__()

        self.transforms = transforms
        self.target_transforms = target_transforms
        assert(name == "shapes")

        shape1, shape2, shape3 = gain_verf_dataset(data_dir, name)

        # shape1 = shape1[:shape1.shape[0] // 10]
        # shape2 = shape2[:shape2.shape[0] // 10]
        # shape3 = shape3[:shape3.shape[0] // 10]
        
        data_shape = shape1.shape
        assert(len(data_shape) == 3)

        train_data_i = np.zeros((data_shape[0] // 2, 3, data_shape[1], data_shape[2]))
        train_data_j = np.zeros((data_shape[0] // 2, 3, data_shape[1], data_shape[2]))
        idx1 = idx2 = idx3 = 0

        for i in range(shape1.shape[0] // 2):
            train_data_i[i, 0, :, :] = shape1[idx1]
            idx1 += 1
            train_data_j[i, 0, :, :] = shape1[idx1]
            idx1 += 1

            train_data_i[i, 1, :, :] = shape2[idx2]
            idx2 += 1
            train_data_j[i, 1, :, :] = shape2[idx2]
            idx2 += 1

            train_data_i[i, 2, :, :] = shape3[idx3]
            idx3 += 1
            train_data_j[i, 2, :, :] = shape3[idx3]
            idx3 += 1

        assert(train == True)
        assert(ver == True)

        # split train and verification dataset
        data_size = int(train_data_i.shape[0] * train_ver_splie_rate)
        if is_ver:
            self.data_i = train_data_i[data_size:, :]
            self.data_j = train_data_j[data_size:, :]
        else:
            self.data_i = train_data_i[:data_size, :]
            self.data_j = train_data_j[:data_size, :]
    
        self.data_i = torch.tensor(self.data_i, dtype=torch.float32)
        self.data_j = torch.tensor(self.data_j, dtype=torch.float32)

        print(self.data_i.shape)
        print(self.data_j.shape)

    def __getitem__(self, idx):
        data1, data2 = self.data_i[idx], self.data_j[idx]
        
        if self.transforms:
            data1 = self.transforms(data1)
            data2 = self.transforms(data2)
        
        return data1, data2

    def __len__(self):
        return self.data_i.shape[0]


class ClrNetBatchDataset(Dataset):
    def __init__(self, data_dir, name, is_single=True, train=True, ver=False, is_ver=False
                    , train_ver_splie_rate=0.9, transforms=None, target_transforms=None, shape_id=0):
        """
        Init torch dataset.
        data_dir: directory to the dataset h5 files.
        name: name of the data, eg: bars.
        is_single: whether the image contains only one object. (Default True)
        train: load train dataset or val. (Default True)

        split_train_ver: whether split train verification test. (Default False)
        is_val: this is verification test. (default False)
        train_ver_splie_rate: split rate of train and verification test. (Default 0.9)
        """

        super(ClrNetBatchDataset, self).__init__()

        self.transforms = transforms
        self.target_transforms = target_transforms
        assert(name == "shapes")

        shape1, shape2, shape3 = gain_verf_dataset(data_dir, name)

        if shape_id == 0:
            self.shape_chosen = np.array(shape1)
        elif shape_id == 1:
            self.shape_chosen = np.array(shape2)
        elif shape_id == 2:
            self.shape_chosen = np.array(shape3)
        else:
            print("Error shape ID!")
            return -1

        data_shape = shape1.shape
        assert(len(data_shape) == 3)

        assert(train == True)
        assert(ver == True)

        # split train and verification dataset
        data_size = int(self.shape_chosen.shape[0] * train_ver_splie_rate)
        if is_ver:
            self.data = self.shape_chosen[data_size:, :]
        else:
            self.data = self.shape_chosen[:data_size, :]
    
        self.data = torch.tensor(self.data, dtype=torch.float32)

        print(self.data.shape)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        
        if self.transforms:
            data_ = self.transforms(data_)
        
        return data_

    def __len__(self):
        return self.data.shape[0]


def open_dataset(data_dir, name):
    """
    open dataset files.
    data_dir: directory to the dataset h5 files
    name: name of the data, eg: bars
    """
    filename = os.path.join(data_dir, name + '.h5')
    return h5py.File(filename, 'r')


def gain_dataset(data_dir, name):
    if name == "clevr":
        imgs = np.zeros((1000, 3, 60, 60))
        for i in range(1000):
            img_idx = '%06d' % (i)
            img = mpimg.imread('./clevr/single_clevr/CLEVR_new_' + img_idx + '.png')
            img = img[:, :, :3]
            img = img.transpose(2, 0, 1)
            imgs[i, :] = img
        imgs_multi = np.zeros((2000, 3, 60, 60))
        for i in range(1000):
            img_idx = '%06d' % (i)
            img = mpimg.imread('./clevr/single_clevr/CLEVR_new_' + img_idx + '.png')
            img = img[:, :, :3]
            img = img.transpose(2, 0, 1)
            imgs_multi[i, :] = img
        return imgs, imgs_multi
    with open_dataset(data_dir, name) as f:
        train_multi_data = f["train_multi"]["default"][:]
        train_single_data = f["train_single"]["default"][:]
        test_data = f["test"]["default"][:]

        train_multi_label = f["train_multi"]["groups"][:]
        train_single_label = f["train_single"]["groups"][:]
        test_label = f["test"]["groups"][:]

    return train_single_data, train_multi_data, test_data, train_single_label, train_multi_label, test_label


def gain_verf_dataset(data_dir, name):
    """
    只保存含有一个物体的数据集
    """
    if name == "bars":
        with open_dataset(data_dir, name) as f:
            hori = f["train_single_hori"]["default"][:]
            vert = f["train_single_vert"]["default"][:]
        return hori, vert
    elif name == "shapes":
        with open_dataset(data_dir, name) as f:
            s0 = f["train_single_0"]["default"][:]
            s1 = f["train_single_1"]["default"][:]
            s2 = f["train_single_2"]["default"][:]
        return s0, s1, s2


def test_bars_dataset():
    with open_dataset("./tmp_data", "bars") as f:
        print(f["train_multi"]["default"][:, 0])
        print(f["train_multi"]["groups"][:, 0])
    bars_dataset = BindingDataset("./tmp_data", "bars", is_single=False, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=bars_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print("img shape", img.shape)
            print(label.shape)
            break
        break


def test_corners_dataset():
    with open_dataset("./tmp_data", "corners") as f:
        print(f["train_multi"]["default"][:, 0])
        print(f["train_multi"]["groups"][:, 0])
    corners_dataset = BindingDataset("./tmp_data", "corners", is_single=False, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=corners_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print(img.shape)
            print(label.shape)
            break
        break


def test_verf(data_name):
    if data_name == "bars":
        hori, vert = gain_verf_dataset("./tmp_data", data_name)
        print(hori[0], vert[0])
    if data_name == "shapes":
        s0, s1, s2 = gain_verf_dataset("./tmp_data", data_name)
        print(s0[0], s1[0], s2[0])


def test_clevr_dataset():
    # 1. 直接获取clevr中single data数据的方法：
    single_clevr_data, multi_clevr_data = gain_dataset("", "clevr")
    print(single_clevr_data.shape, multi_clevr_data.shape)

    # 2.1 使用pytorch中的dataloader加载single object clevr的方法：
    bars_dataset = BindingDataset("", "clevr", is_single=True, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=bars_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print(img.shape)
            print(label.shape)
            break
        break

    # 3. 使用pytorch中的dataloader加载multiple objects clevr的方法：
    bars_dataset = BindingDataset("", "clevr", is_single=False, train=True, ver=True, is_ver=False)
    train_loader = DataLoader(dataset=bars_dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(32):
        for iter, data in enumerate(train_loader):
            img, label = data
            print(img.shape)
            print(label.shape)
            break
        break


if __name__ == "__main__":
    # test_corners_dataset()
    # test_verf("shapes")
    # test_clevr_dataset()
    test_bars_dataset()
