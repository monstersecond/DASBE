#!/bin/bash

cd dasbe
if [ ! -d "./tmp_imgs" ];then
    mkdir ./tmp_imgs
fi
if [ ! -d "./tmp_imgs/clrnet_batch" ];then
    mkdir ./tmp_imgs/clrnet_batch
fi
python train_clrnet.py

./figure_code/draw_figure4.sh

cd ..
