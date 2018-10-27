from collections import Counter
from typing import Mapping, List, Tuple

from matplotlib import pyplot as plt
import numpy as np

from bartpy.model import Model
from bartpy.sklearnmodel import SklearnModel


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


def null_feature_split_proportions_distribution(model: SklearnModel,
                                                X: np.ndarray,
                                                y: np.ndarray,
                                                n_permutations: int=10) -> Mapping[int, List[float]]:
    """
    Calculate a null distribution of proportion of splits on each variable in X

    Works by randomly permuting y to remove any true dependence of y on X and calculating feature importance

    Parameters
    ----------
    model: SklearnModel
        Model specification to work with
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target data
    n_permutations: int
        How many permutations to run
        The higher the number of permutations, the more accurate the null distribution, but the longer it will take to run
    Returns
    -------
    Mapping[int, List[float]]
        A list of inclusion proportions for each variable in X
    """

    inclusion_dict = {x: [] for x in range(X.shape[1])}

    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        model.fit(X, y_perm)
        splits_perm = feature_split_proportions_counter(model)
        for key, value in splits_perm:
            inclusion_dict[key].append(value)

    return inclusion_dict