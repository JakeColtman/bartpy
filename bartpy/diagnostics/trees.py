from typing import List

from matplotlib import pyplot as plt
import numpy as np

from bartpy.model import Model


def plot_tree_depth(model_samples: List[Model]):
    min_depth, mean_depth, max_depth = [], [], []
    for sample in model_samples:
        model_depths = []
        for tree in sample.trees:
            print(tree.nodes[0].depth)
            model_depths += [x.depth for x in tree.nodes]
        min_depth.append(np.min(model_depths))
        mean_depth.append(np.mean(model_depths))
        max_depth.append(np.max(model_depths))

    plt.plot(min_depth)
    plt.plot(mean_depth)
    plt.plot(max_depth)
    plt.show()