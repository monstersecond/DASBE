#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import seaborn as sns


def plot_groups(groups, ax):
    mask = (groups == 0)
    sns.heatmap(groups, mask=mask, square=True, cmap='viridis_r',
                xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    #sns.heatmap(groups, square=True, cmap='viridis_r',
                # xticklabels=False, yticklabels=False, cbar=False, ax=ax)


def plot_input_image(img, ax):
    sns.heatmap(img, square=True, xticklabels=False,
                yticklabels=False, cmap='Greys', cbar=False, ax=ax)

