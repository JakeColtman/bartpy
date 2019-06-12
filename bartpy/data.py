from operator import le, gt
from typing import Any, List, Union

import numpy as np
import pandas as pd

from bartpy.errors import NoSplittableVariableException
from bartpy.splitcondition import SplitCondition


def is_not_constant(series: np.ma.masked_array) -> bool:
    """
    Quickly identify whether a series contains more than 1 distinct value
    Parameters
    ----------
    series: np.ndarray
    The series to assess

    Returns
    -------
    bool
        True if more than one distinct value found
    """
    if len(series) <= 1:
        return False
    first_value = None
    for i in range(1, len(series)):
        if not series.mask[i] and series.data[i] != first_value:
            if first_value is None:
                first_value = series.data[i]
            else:
                return True
    return False


def ensure_numpy_array(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
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
    X = format_covariate_matrix(X)
    y = y.astype(float)
    return Data(X, y, np.zeros_like(X).astype(bool), normalize)


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
                 mask: np.ndarray=None,
                 normalize=False,
                 unique_columns=None):
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        if mask is None:
            mask = np.zeros_like(X).astype(bool)
        self._mask = mask
        self._X = X
        self._masked_X = self._X.view(np.ma.MaskedArray)
        self._masked_X[self._mask] = np.ma.masked
        self._unique_columns = unique_columns

        if normalize:
            self.original_y_min, self.original_y_max = y.min(), y.max()
            self._y = self.normalize_y(y)
        else:
            self._y = y

        self._masked_y = np.ma.masked_array(self._y, self._mask[:,0])
        self.y_cache_up_to_date = True
        self.y_sum_cache_up_to_date = True
        self._summed_y = np.sum(self.y)
        self._max_values_cache = self.X.filled(-np.inf).max(axis=0)
        self._splittable_variables = [x for x in range(0, self.X.shape[1]) if is_not_constant(self.X[:, x])]
        self._n_obsv = int(np.sum(~self.y.mask))

    def summed_y(self):
        if self.y_sum_cache_up_to_date:
            return self._summed_y
        else:
            self._summed_y = np.sum(self.y)
            self.y_sum_cache_up_to_date = True
            return self.summed_y()

    @property
    def unique_columns(self):
        if self._unique_columns is None:
            unique_columns = []
            for i in range(self._X.shape[1]):
                if len(np.unique(self._X[:, i])) == len(self._X):
                    unique_columns.append(i)
            self._unique_columns = unique_columns
        return self._unique_columns

    @property
    def y(self) -> np.ma.masked_array:
        if self.y_cache_up_to_date:
            return self._masked_y
        else:
            self._masked_y = np.ma.masked_array(self._y, mask=self.mask[:,0])
            self.y_cache_up_to_date = True
            return self._masked_y

    @property
    def X(self) -> np.ma.masked_array:
        return self._masked_X

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def splittable_variables(self) -> List[int]:
        """
        List of columns that can be split on, i.e. that have more than one unique value

        Returns
        -------
        List[int]
            List of column numbers that can be split on
        """
        return self._splittable_variables

    @property
    def variables(self) -> List[int]:
        """
        The set of variable names the data contains.
        Of dimensionality p

        Returns
        -------
        List[int]
        """
        return list(range(0, self._X.shape[1]))

    def random_splittable_variable(self) -> str:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        splittable_variables = list(self.splittable_variables())
        if len(splittable_variables) == 0:
            raise NoSplittableVariableException()
        return np.random.choice(np.array(list(splittable_variables)), 1)[0]

    def random_splittable_value(self, variable: int) -> Any:
        """
        Return a random value of a variable
        Useful for choosing a variable to split on

        Parameters
        ----------
        variable - str
            Name of the variable to split on

        Returns
        -------
        Any

        Notes
        -----
          - Won't create degenerate splits, all splits will have at least one row on both sides of the split
        """
        if variable not in self._splittable_variables:
            return None
        max_value = self._max_values_cache[variable]
        candidate = np.random.choice(self.X[:, variable])
        while candidate == max_value:
            candidate = np.random.choice(self.X[:, variable])
        return candidate

    @property
    def n_obsv(self) -> int:
        return self._n_obsv

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        if variable in self.unique_columns:
            n_obsv = self.n_obsv
            if n_obsv == 0.:
                return 0.
            return 1. / n_obsv
        else:
            return float(np.mean(self._X[:, variable] == value))

    @staticmethod
    def normalize_y(y: np.ndarray) -> np.ndarray:
        """
        Normalize y into the range (-0.5, 0.5)
        Useful for allowing the leaf parameter prior to be 0, and to standardize the sigma prior

        Parameters
        ----------
        y - np.ndarray

        Returns
        -------
        np.ndarray

        Examples
        --------
        >>> Data.normalize_y([1, 2, 3])
        array([-0.5,  0. ,  0.5])
        """
        y_min, y_max = np.min(y), np.max(y)
        return -0.5 + ((y - y_min) / (y_max - y_min))

    def unnormalize_y(self, y: np.ndarray) -> np.ndarray:
        distance_from_min = y - (-0.5)
        total_distance = (self.original_y_max - self.original_y_min)
        return self.original_y_min + (distance_from_min * total_distance)

    @property
    def unnormalized_y(self) -> np.ndarray:
        return self.unnormalize_y(self.y)

    @property
    def normalizing_scale(self) -> float:
        return self.original_y_max - self.original_y_min

    def update_y(self, y):
        self._y = y
        self.y_cache_up_to_date = False
        self.y_sum_cache_up_to_date = False

    def _update_mask(self, other: SplitCondition):
        if other.operator == gt:
            column_mask = self._X[:, other.splitting_variable] <= other.splitting_value
        elif other.operator == le:
            column_mask = self._X[:, other.splitting_variable] > other.splitting_value
        else:
            raise TypeError("Operator type not matched, only {} and {} supported".format(gt, le))

        return self.mask | np.tile(column_mask, (self.mask.shape[1], 1)).T

    def __add__(self, other: SplitCondition):
        updated_mask = self._update_mask(other)

        return Data(self._X,
                    self._y,
                    updated_mask,
                    normalize=False,
                    unique_columns=None)
