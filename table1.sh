#!/bin/bash

cd dasbe

if [ ! -d "./tmp_net" ];then
    mkdir ./tmp_net
fi

# First, train networks
# ./train_nets.sh

./table1/unified_clustering.sh

cd ..
