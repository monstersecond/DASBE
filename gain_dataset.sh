#!/bin/bash

cd dasbe

if [ ! -d "./tmp_data" ];then
    mkdir ./tmp_data
fi

python ./dataset/bars.py
echo "FINISH bars"

python ./dataset/corner.py
echo "FINISH corner"

python ./dataset/mnist.py
echo "FINISH mnist"

python ./dataset/mnist_shape.py
echo "FINISH mnist_shape"

python ./dataset/moving_shapes.py
echo "FINISH moving_shapes"

python ./dataset/moving_shapes2.py
echo "FINISH moving_shapes2"

python ./dataset/multi_mnist.py
echo "FINISH multi_mnist"

python ./dataset/shapes_with_pos.py
echo "FINISH shapes_with_pos"

python ./dataset/shapes.py
echo "FINISH shapes"

cd ..
