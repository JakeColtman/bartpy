from operator import le, gt
from typing import List

import numpy as np

from bartpy.errors import NoSplittableVariableException
from bartpy.splitcondition import SplitCondition


class CovariateMatrix(object):
    """
    Encapsualtes the features of the model at a particular node in the decision tree

    Provides a clean interface for pulling information about the matrix e.g.
      - which columns can be split on
      - how many unique values each column has

    Additionally, provides aggressive caching to help performance
    """

    def __init__(self,
                 X: np.ndarray,
                 mask: np.ndarray,
                 n_obsv: int,
                 unique_columns: List[int],
                 splittable_variables: List[int]):

        self._X = X
        self._n_obsv = n_obsv
        self._n_features = X.shape[1]
        self._mask = mask

        # Cache iniialization
        if unique_columns is not None:
            self._unique_columns = [x if x is True else None for x in unique_columns]
        else:
            self._unique_columns = [None for _ in range(self._n_features)]
        if splittable_variables is not None:
            self._splittable_variables = [x if x is False else None for x in splittable_variables]
        else:
            self._splittable_variables = [None for _ in range(self._n_features)]
        self._max_values = [None] * self._n_features
        self._X_column_cache = [None] * self._n_features
        self._max_value_cache = [None] * self._n_features
        self._X_cache = None

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def values(self) -> np.ndarray:
        return self._X

    def get_column(self, i: int) -> np.ndarray:
        if self._X_cache is None:
            self._X_cache = self.values[~self.mask, :]
        return self._X_cache[:, i]

    def splittable_variables(self) -> List[int]:
        """
        List of columns that can be split on, i.e. that have more than one unique value

        Returns
        -------
        List[int]
            List of column numbers that can be split on
        """
        for i in range(0, self._n_features):
            if self._splittable_variables[i] is None:
                self._splittable_variables[i] = is_not_constant(self.get_column(i))
        
        return [i for (i, x) in enumerate(self._splittable_variables) if x is True]        

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    def is_at_least_one_splittable_variable(self) -> bool:
        if any(self._splittable_variables):
            return True
        else:
            return len(self.splittable_variables()) > 0
    
    def random_splittable_variable(self) -> str:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        if self.is_at_least_one_splittable_variable():
            return np.random.choice(np.array(self.splittable_variables()), 1)[0]
        else:
            raise NoSplittableVariableException()

    def is_column_unique(self, i: int) -> bool:
        """
        Identify whether feature contains only unique values, i.e. it has no duplicated values
        Useful to provide a faster way to calculate the probability of a value being selected in a variable

        Returns
        -------
        List[int]
        """
        if self._unique_columns[i] is None:
            self._unique_columns[i] = len(np.unique(self.get_column(i))) == self._n_obsv
        return self._unique_columns[i]

    def max_value_of_column(self, i: int):
        if self._max_value_cache[i] is None:
            self._max_value_cache[i] = self.get_column(i).max()
        return self._max_value_cache[i]

    def random_splittable_value(self, variable: int) -> float:
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
        if variable not in self.splittable_variables():
            raise NoSplittableVariableException()
        max_value = self.max_value_of_column(variable)
        candidate = np.random.choice(self.get_column(variable))
        while candidate == max_value:
            candidate = np.random.choice(self.get_column(variable))
        return candidate

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        if self.is_column_unique(variable):
            return 1. / self.n_obsv
        else:
            return float(np.mean(self.get_column(variable) == value))

    def update_mask(self, other: SplitCondition) -> np.ndarray:
        if other.operator == gt:
            column_mask = self.values[:, other.splitting_variable] <= other.splitting_value
        elif other.operator == le:
            column_mask = self.values[:, other.splitting_variable] > other.splitting_value
        else:
            raise TypeError("Operator type not matched, only {} and {} supported".format(gt, le))

        return self.mask | column_mask

    @property
    def variables(self) -> List[int]:
        return list(range(self._n_features))

    @property
    def n_obsv(self) -> int:
        return self._n_obsv


def is_not_constant(series: np.ndarray) -> bool:
    """
    Quickly identify whether a series contains more than 1 distinct value.

    Parameters
    ----------
    series: np.ndarray
    The series to assess

    Returns
    -------
    bool
        True if more than one distinct value found

    Examples
    --------
    >>> is_not_constant(np.array([1, 1, 1]))
    False
    >>> is_not_constant(np.array([1, 2, 3]))
    True
    >>> is_not_constant(np.array([]))
    False

    """
    if len(series) <= 1:
        return False
    first_value = series[0]
    for i in range(1, len(series)):
        if series[i] != first_value:
            return True
    return False


if __name__ == "__main__":
    import doctest
    doctest.testmod()
