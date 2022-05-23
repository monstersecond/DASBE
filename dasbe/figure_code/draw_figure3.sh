#!/bin/bash

cd ./table1/unified_clustering/
python syn_rate_score_all.py
cd ../..

cd ./figure_code/fig5
python draw_fig5.py
cd ../../
