#!/bin/bash

cd ./table1/unified_clustering
if [ ! -d "./AMI_npys" ];then
    mkdir ./AMI_npys
fi

python clustering_dynamics_all.py

python PCNN_dynamics_all_torch.py

python pure_ANN_clustering_all.py

python table1.py

cd ../../
