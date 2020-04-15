from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bartpy.covariates import CovariateMatrix, DataFrame
from bartpy.splitcondition import SplitCondition
from bartpy.target import Target
from bartpy.samplers.scalar import VariableWidthDiscreteSampler


def ensure_numpy_array(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    else:
        return X


def ensure_float_array(X: np.ndarray) -> np.ndarray:
    return X.astype(float)


def format_covariate_matrix(X: DataFrame) -> np.ndarray:
    X = ensure_numpy_array(X)
    return ensure_float_array(X)


def make_bartpy_data(X: DataFrame,
                     y: np.ndarray,
                     normalize: bool=True) -> 'Data':
    X = format_covariate_matrix(X)
    y = y.astype(float)
    return Data(X, y, normalize=normalize)


class Data(object):
    """
    Encapsulates the data within a split of feature space.
    Primarily used to cache computations on the data for better performance

    Parameters
    ----------
    X: np.ndarray
        The subset of the covariate matrix that falls into the split
    y: np.ndarray
        The subset of the target array that falls into the split
    normalize: bool
        Whether to map the target into -0.5, 0.5
    cache: bool
        Whether to cache common values.
        You really only want to turn this off if you're not going to the resulting object for anything (e.g. when testing)
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 mask: Optional[np.ndarray]=None,
                 normalize: bool=False,
                 unique_columns: List[Optional[bool]]=None,
                 splittable_variables: Optional[List[Optional[bool]]]=None,
                 choice_sampler: VariableWidthDiscreteSampler=None):

        if mask is None:
            mask = np.ones_like(y).astype(bool)

        if choice_sampler is None:
            choice_sampler = VariableWidthDiscreteSampler()
        self.choice_sampler = choice_sampler

        self._mask: DataFrame = mask

        n_obsv = (self.mask).astype(int).sum()

        self._n_obsv = n_obsv

        self._X = CovariateMatrix(X, mask, n_obsv, unique_columns=unique_columns, splittable_variables=splittable_variables, choice_sampler=choice_sampler)
        self._y = Target(y, mask, n_obsv, normalize)

    @property
    def y(self) -> Target:
        return self._y

    @property
    def X(self) -> CovariateMatrix:
        return self._X

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def update_y(self, y: np.ndarray) -> None:
        self._y.update_y(y)

    def __add__(self, other: SplitCondition) -> 'Data':
        updated_mask = self.X.update_mask(other)

        return Data(self.X.values,
                    self.y.values,
                    updated_mask,
                    normalize=False,
                    unique_columns=self._X.unique_variables_cache,
                    splittable_variables=self._X.splittable_variables_cache,
                    choice_sampler=self.choice_sampler)
