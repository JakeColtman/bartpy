from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bartpy.data.covariatematrix import CovariateMatrix
from bartpy.data.target import Target
from bartpy.splitcondition import SplitCondition


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
                 unique_columns: List[int]=None,
                 splittable_variables: Optional[List[Optional[bool]]]=None,
                 y_sum: float=None,
                 n_obsv: int=None):

        if mask is None:
            mask = np.zeros_like(y).astype(bool)
        self._mask: np.ndarray = mask

        if n_obsv is None:
            n_obsv = (~self.mask).astype(int).sum()

        self._n_obsv = n_obsv

        self._X = CovariateMatrix(X, mask, n_obsv, unique_columns, splittable_variables)
        self._y = Target(y, mask, n_obsv, normalize, y_sum)

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
                    unique_columns=self._X._unique_columns,
                    splittable_variables=self._X._splittable_variables,
                    y_sum=other.carry_y_sum,
                    n_obsv=other.carry_n_obsv)

def ensure_numpy_array(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Ensure that the data passed to bartpy is a numpy ndarray

    Examples
    --------
    >>> X = np.array([[1.0], [2.0]])
    >>> ensure_numpy_array(pd.DataFrame(X, columns=["a"]))
    array([[1.],
           [2.]])
    >>> ensure_numpy_array(X)
    array([[1.],
           [2.]])
    """

    if isinstance(X, pd.DataFrame):
        return X.values
    else:
        return X


def ensure_float_array(X: np.ndarray) -> np.ndarray:
    return X.astype(float)


def format_covariate_matrix(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    X = ensure_numpy_array(X)
    return ensure_float_array(X)


def make_bartpy_data(X: Union[np.ndarray, pd.DataFrame],
                     y: np.ndarray,
                     normalize: bool=True) -> 'Data':
    """
    Convert data into a format that can be consumed by bartpy
    Handles:
      - enuring everything is a float
      - wrapping data into CovariateMatrix and Target objects
      - normalizing target

    Parameters
    ----------
    X: Union[np.ndarray, pd.DataFrame]
        Matrix of features
    y: np.ndarray
        Array of target
    normalize: bool
        Whether to map the target into the range (-0.5, 0.5)
    """
    X = format_covariate_matrix(X)
    y = y.astype(float)
    return Data(X, y, normalize=normalize)


if __name__ == "__main__":
    import doctest
    doctest.testmod()