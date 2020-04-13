from abc import ABCMeta
from operator import gt, le
from typing import List, Any, Union, Optional

import numpy as np
import pandas as pd
import torch

from bartpy.errors import NoSplittableVariableException
from bartpy.splitcondition import SplitCondition
from bartpy.samplers.scalar import VariableWidthDiscreteSampler


DataFrame = Union[np.ndarray, pd.DataFrame, torch.Tensor]


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
    if len(series) <= 1:
        return False
    first_value = None
    for i in range(1, len(series)):
        if series[i] != first_value:
            if first_value is None:
                first_value = series.data[i]
            else:
                return True
    return False


class CovariateMatrix(object, metaclass=ABCMeta):

    def __init__(self,
                 X: DataFrame,
                 mask: np.ndarray,
                 n_obsv: int,
                 unique_columns: List[Optional[bool]],
                 splittable_variables: List[Optional[bool]],
                 choice_sampler: VariableWidthDiscreteSampler):

        self._X = X
        self._n_obsv = n_obsv
        self._n_features = X.shape[1]
        self._mask = mask
        self.choice_sampler = choice_sampler

        # Cache initialization
        if unique_columns is not None:
            self._unique_columns = [x if x else None for x in unique_columns]
        else:
            self._unique_columns = [None for _ in range(self._n_features)]

        if splittable_variables is not None:
            self._splittable_variables = [x if x is False else None for x in splittable_variables]
        else:
            self._splittable_variables = [None for _ in range(self._n_features)]

        self._max_value_cache = [None] * self._n_features
        self._X_cache = None

    def get_column(self, i: int) -> np.ndarray:
        if self._X_cache is None:
            if isinstance(self.values, torch.Tensor):
                values = self.values.numpy()
            else:
                values = self.values

            filtered_values = values[self.mask, :]
            if isinstance(self.values, torch.Tensor):
                filtered_values = torch.from_numpy(filtered_values)

            self._X_cache = filtered_values
        return self._X_cache[:, i]

    def is_variable_splittable(self, i: int) -> bool:
        if self.is_column_unique(i) and self.n_obsv > 1:
            return True
        else:
            return is_not_constant(self.get_column(i))

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
                self._splittable_variables[i] = self.is_variable_splittable(i)

        return [i for (i, x) in enumerate(self._splittable_variables) if x]

    def is_at_least_one_splittable_variable(self) -> bool:
        if any(self.splittable_variables_cache):
            return True
        else:
            return len(self.splittable_variables()) > 0

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
        max_value = self.max_value_of_column(variable)
        candidate = self.choice_sampler.sample(self.get_column(variable))
        while candidate == max_value:
            candidate = self.choice_sampler.sample(self.get_column(variable))
        return candidate

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
        return float(self._max_value_cache[i])

    def random_splittable_variable(self) -> int:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        if self.is_at_least_one_splittable_variable():
            return self.choice_sampler.sample(self.splittable_variables())
        else:
            raise NoSplittableVariableException()

    def proportion_of_value_in_variable(self, variable: int, value: float) -> float:
        if self.is_column_unique(variable):
            return 1. / self.n_obsv
        else:
            return float((self.get_column(variable) == value).mean())

    def update_mask(self, other: SplitCondition) -> np.ndarray:
        if other.operator == gt:
            column_mask = self.values[:, other.splitting_variable] > other.splitting_value
        elif other.operator == le:
            column_mask = self.values[:, other.splitting_variable] <= other.splitting_value
        else:
            raise TypeError("Operator type not matched, only {} and {} supported".format(gt, le))

        return self.mask & column_mask

    @property
    def variables(self) -> List[int]:
        return list(range(self._n_features))

    @property
    def n_obsv(self) -> int:
        return self._n_obsv

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    @property
    def values(self) -> np.ndarray:
        return self._X

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def splittable_variables_cache(self) -> List[Optional[bool]]:
        return self._splittable_variables

    @property
    def unique_variables_cache(self) -> List[Optional[bool]]:
        return self._unique_columns