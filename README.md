# Dance of SNN and ANN: Solving binding problem by combining spike timing and reconstructive attentio

This repository is the code for the paper *Dance of SNN and ANN: Solving binding problem by combining spike timing and reconstructive attention.* It can reproduce the main results of the paper. 

The binding problem is one of the fundamental challenges that prevent the artificial neural network (ANNs) from a compositional understanding of the world like human perception, because disentangled and distributed representations of generative factors can interfere and lead to ambiguity when complex data with multiple objects are presented. In this paper, we propose a brain-inspired unsupervised hybrid neural network (HNN) that introduces temporal binding theory originated from neuroscience into ANNs by integrating spike timing dynamics (via spiking neural networks, SNNs) with reconstructive attention (by ANNs). Spike timing provides an additional dimension for grouping, while reconstructive feedback coordinates the spikes into temporal coherent states. Through iterative interaction of ANN and SNN, the model continuously binds multiple objects at alternative synchronous firing times in the SNN coding space. The effectiveness of the model is evaluated on five artificially generated datasets of binary images. By visualization and analysis, we demonstrate that the binding is explainable, soft, flexible, and hierarchical. Notably, the model is trained on single object datasets without explicit supervision on grouping, but can successfully bind multiple objects on test datasets, showing its compositional generalization capability. Further results show its binding ability in dynamic situations.

## Preparation

The following packages should be installed before running this project:

```
pytorch
matplotlib
seaborn
h5py
numpy

```

## Gain all of the datasets for all experiments

Running the following script can generate all files of dataset for the experiments showing in the paper.

```
./gain_dataset.sh
```

## Train networks

Running the following script can train all networks for the experiments except moving dataset.

```
./train_nets.sh
```

## Code for figures in the paper

The following scripts can run the experiments for figure 2, figure 3, figure 4, and figure 5.

```
./figure2.sh
./figure3.sh
./figure4.sh
./figure5.sh
```

For parts of the results in figure 2 and figure 4, the results can be found in the notebook as follows:

```
./dasbe/figure_code/draw_figure2.ipynb
./dasbe/figure_code/draw_figure4.ipynb
```

## Code for table I in the paper

The following script can generate the experiment results in table I.

```
./table1.sh
```

## An easy example to show the binding results of DASBE

The file `dasbe_example.py` shows an example of using the code in `dasbe` library.

Running the script `dasbe_example.sh` will call `dasbe_example.py` with appropriate parameters.
