from copy import deepcopy
from typing import List, Mapping, Union, Tuple

from joblib import Parallel
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from bartpy.sklearnmodel import SklearnModel


def original_model_rmse(model: SklearnModel,
                        X: Union[pd.DataFrame, np.ndarray],
                        y: np.ndarray,
                        n_k_fold_splits: int) -> List[float]:

    kf = KFold(n_k_fold_splits, shuffle=True)

    base_line_rmses = []

    for train_index, test_index in kf.split(X):
        model = deepcopy(model)
        model.fit(X[train_index], y[train_index])
        base_line_rmses.append(model.rmse(X[test_index], y[test_index]))

    return base_line_rmses


def null_rmse_distribution(model: SklearnModel,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: np.ndarray,
                           variable: int,
                           n_k_fold_splits: int,
                           n_permutations: int=10) -> List[float]:
    """
    Calculate a null distribution on the RMSEs after scrambling a variable

    Works by randomly permuting y to remove any true dependence of y on X and calculating feature importance

    Parameters
    ----------
    model: SklearnModel
        Model specification to work with
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target data
    variable: int
        Which column of the covariate matrix to scramble
    n_permutations: int
        How many permutations to run
        The higher the number of permutations, the more accurate the null distribution, but the longer it will take to run
    Returns
    -------
    Mapping[int, List[float]]
        A list of inclusion proportions for each variable in X
    """

    kf = KFold(n_k_fold_splits, shuffle=True)

    null_rmses = []

    for train_index, test_index in kf.split(X):
        delayed_chains = []
        for _ in range(n_permutations):
            permuted_model = deepcopy(model)
            permuted_X = deepcopy(X)
            permuted_X[:, variable] = np.random.permutation(permuted_X[:, variable])
            delayed_chains += permuted_model.delayed_chains(permuted_X[train_index], y[train_index])

            combined_extracts = Parallel(model.n_jobs)(delayed_chains)
            for extract in combined_extracts:
                extracted_model = model.from_extract(extract)
                null_rmses.append(extracted_model.rmse(permuted_X[test_index], y[test_index]))

    return null_rmses


def feature_importance(model: SklearnModel,
                       X: Union[pd.DataFrame, np.ndarray],
                       y: np.ndarray,
                       variable: int,
                       n_k_fold_splits: int=2,
                       n_permutations: int=10) -> Tuple[List[float], List[float]]:
    original_model = original_model_rmse(model, X, y, n_k_fold_splits)
    null_distribution = null_rmse_distribution(model, X, y, variable, n_k_fold_splits, n_permutations)

    plt.hist(null_distribution, label="Null Distribution")
    plt.hist(original_model, label="Original Model")
    plt.title("RMSE of full model against null distribution for variable {}".format(variable))
    plt.xlabel("RMSE")
    plt.ylabel("Density")

    return original_model, null_distribution