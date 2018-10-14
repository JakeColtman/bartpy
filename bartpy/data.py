from collections import namedtuple
from typing import Any, List

import numpy as np
import pandas as pd

from bartpy.errors import NoSplittableVariableException


SplitData = namedtuple("SplitData", ["left_data", "right_data"])


def is_not_constant(series: np.ndarray) -> bool:
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
    if len(series) == 1:
        return False
    start_value = series[0]
    for val in series[1:]:
        if val != start_value:
            return True
    return False


class Data:
    """
    Encapsulates the data within a split of feature space.
    Primarily used to cache computations on the data for better performance

    Parameters
    ----------
    X: np.ndarray
        The subset of the covariate matrix that falls into the split
    y: np.ndarry
        The subset of the target array that falls into the split
    normalize: bool
        Whether to map the target into -0.5, 0.5
    cache: bool
        Whether to cache common values.
        You really only want to turn this off if you're not going to the resulting object for anything (e.g. when testing)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, normalize=False, cache=True, unique_columns=None):
        if type(X) == pd.DataFrame:
            X = X.values
        self._X = X
        self._unique_columns = unique_columns

        if normalize:
            self.original_y_min, self.original_y_max = y.min(), y.max()
            self._y = self.normalize_y(y)
        else:
            self._y = y

        if cache:
            self._max_values_cache = self._X.max(axis=0)
            self._splittable_variables = [x for x in range(0, self._X.shape[1]) if is_not_constant(self._X[:, x])]
            self._n_unique_values_cache = [None] * self._X.shape[1]

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
    def y(self) -> np.ndarray:
        return self._y

    @property
    def X(self) -> np.ndarray:
        return self._X

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
        return len(self.X)

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        if variable in self.unique_columns:
            return 1. / self.n_obsv
        else:
            return np.mean(self._X[:, variable] == value)

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