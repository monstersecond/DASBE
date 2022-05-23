#!/bin/bash

cd dasbe

if [ ! -d "./tmp_net" ];then
    mkdir ./tmp_net
fi
if [ ! -d "./tmp_imgs" ];then
    mkdir ./tmp_imgs
fi

# First, train_networks
# 1. bars
if [ ! -d "./tmp_imgs/bar" ];then
    mkdir ./tmp_imgs/bar/
fi
python train_net.py bars

# 2. corners
if [ ! -d "./tmp_imgs/corner" ];then
    mkdir ./tmp_imgs/corner
fi
python train_net.py corner  

# 3. shapes
if [ ! -d "./tmp_imgs/shape" ];then
    mkdir ./tmp_imgs/shape
fi
python train_net.py shape

# 4. multi-MNIST
if [ ! -d "./tmp_imgs/multi_mnist" ];then
    mkdir ./tmp_imgs/multi_mnist
fi
train_multi_mnist.py

# 5. MNIST+Shape
if [ ! -d "./tmp_imgs/shape" ];then
    mkdir ./tmp_imgs/shape
fi
python train_net.py mnist_shape  

cd ..

