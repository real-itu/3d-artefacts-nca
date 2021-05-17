import copy

import matplotlib.pyplot as plt
import numpy as np


def voxel_to_numeric(voxels):
    voxels = copy.deepcopy(voxels)
    voxels[voxels == None] = "_empty"
    unique = sorted(np.unique(voxels))
    d = {}
    for i in range(len(unique)):
        voxels[voxels == unique[i]] = i
        d[i] = unique[i]
    return voxels, d


def replace_colors(x, color_dict):
    x = x.astype(object)
    for k in color_dict:
        x[x == int(k)] = color_dict[k]
    return x


def plot_voxels(voxels):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.voxels(voxels, facecolors=voxels, edgecolor="k")
    plt.show()
