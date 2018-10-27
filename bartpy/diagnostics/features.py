from collections import Counter
from typing import List, Mapping, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from bartpy.model import Model
from bartpy.sklearnmodel import SklearnModel


def feature_split_proportions_counter(model_samples: List[Model]) -> Mapping[int, float]:

    split_variables = []
    for sample in model_samples:
        for tree in sample.trees:
            for node in tree.nodes:
                splitting_var = node.split.splitting_variable
                split_variables.append(splitting_var)
    return {x[0]: x[1] / len(split_variables) for x in Counter(split_variables).items() if x[0] is not None}


def plot_feature_split_proportions(model_samples: List[Model]):
    proportions = feature_split_proportions_counter(model_samples)

    y_pos = np.arange(len(proportions))
    name, count = proportions.keys(), proportions.values()

    plt.barh(y_pos, count, align='center', alpha=0.5)
    plt.yticks(y_pos, name)
    plt.xlabel('Proportion of all splits')
    plt.show()


def null_feature_split_proportions_distribution(model: SklearnModel,
                                                X: Union[pd.DataFrame, np.ndarray],
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
        splits_perm = feature_split_proportions_counter(model.model_samples)
        for key, value in splits_perm.items():
            inclusion_dict[key].append(value)

    return inclusion_dict