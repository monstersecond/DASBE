#!/bin/bash

cd ./figure_code/fig4

if [ ! -d "./loss some spike" ];then
    mkdir ./loss\ some\ spike
fi
if [ ! -d "./normal" ];then
    mkdir ./normal
fi
if [ ! -d "./varied period" ];then
    mkdir ./varied\ period
fi

python bind_shapes.py

python draw_fig4_score.py

cd ../..
