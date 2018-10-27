from collections import Counter
from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np

from bartpy.model import Model


def feature_split_proportions_counter(model_samples: List[Model]) -> Tuple[int, Counter]:

    split_variables = []
    for sample in model_samples:
        for tree in sample.trees:
            for node in tree.nodes:
                splitting_var = node.split.splitting_variable
                split_variables.append(splitting_var)
    return len(split_variables), Counter(split_variables)


def plot_feature_split_proportions(model_samples: List[Model]):
    total_count, counts = feature_split_proportions_counter(model_samples)

    y_pos = np.arange(len(counts))
    name = counts.keys()
    count = [x / total_count for x in counts.values()]

    plt.barh(y_pos, count, align='center', alpha=0.5)
    plt.yticks(y_pos, name)
    plt.xlabel('Proportion of all splits')
    plt.show()