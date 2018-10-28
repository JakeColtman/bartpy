from typing import List

from matplotlib import pyplot as plt
import numpy as np

from bartpy.model import Model


def plot_tree_depth(model_samples: List[Model], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    min_depth, mean_depth, max_depth = [], [], []
    for sample in model_samples:
        model_depths = []
        for tree in sample.trees:
            model_depths += [x.depth for x in tree.nodes]
        min_depth.append(np.min(model_depths))
        mean_depth.append(np.mean(model_depths))
        max_depth.append(np.max(model_depths))

    ax.plot(min_depth, label="Min Depth")
    ax.plot(mean_depth, label="Mean Depth")
    ax.plot(max_depth, label="Max Depth")
    ax.set_ylabel("Depth")
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_title("Tree Depth by Iteration")