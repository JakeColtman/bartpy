from abc import ABC
from typing import List, Optional
from copy import deepcopy

import pandas as pd
import numpy as np

from bartpy.data import Data


def fancy_bool(x, bool_mask):
    return x[bool_mask]


class SplitCondition(ABC):

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = None

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data):
        if self._condition is None:
            self._condition = data.X[self.splitting_variable] > self.splitting_value
        return self._condition

    def left(self, data):
        return ~self.condition(data)

    def right(self, data):
        return self.condition(data)


class Split:

    def __init__(self, data: Data, split_conditions: List[SplitCondition]=None, combined_condition=None):
        if split_conditions is None:
            split_conditions = []
        self._conditions = split_conditions
        self._data = deepcopy(data)
        self._combined_condition = combined_condition
        self._conditioned_X = pd.DataFrame(fancy_bool(self._data.X.values, np.array(self.condition())), columns=self._data.X.columns)
        self._cache_up_to_date = False
        self._conditioned_data = None

    @property
    def data(self):
        if self._cache_up_to_date:
            return self._conditioned_data
        else:
            self._conditioned_data = Data(self._conditioned_X, fancy_bool(self._data.y.values, np.array(self.condition())))
            self._cache_up_to_date = True
        return self._conditioned_data

    def combined_condition(self, data):
        if len(self._conditions) == 0:
            return np.array([True] * data.n_obsv)
        else:
            return self._combined_condition

    def condition(self, data: Data=None):
        if data is None:
            if self._combined_condition is None:
                self._combined_condition = self.combined_condition(self._data)
            return self._combined_condition
        else:
            return self.combined_condition(data)

    def __add__(self, other: SplitCondition):
        left = Split(self._data, self._conditions + [other], combined_condition=self.condition() & other.left(self._data))
        right = Split(self._data, self._conditions + [other], combined_condition=self.condition() & other.right(self._data))
        return left, right

    def most_recent_split_condition(self) -> Optional[SplitCondition]:
        if len(self._conditions) > 0:
            return self._conditions[-1]
        else:
            return None

    def update_y(self, y):
        self._cache_up_to_date = False
        self._data._y = y


def sample_split_condition(node, variable_prior=None) -> Optional[SplitCondition]:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws
        - draw a node to split on based on multinomial distribution
        - draw an observation within that variable to split on

    Parameters
    ----------
    node - TreeNode
    variable_prior - np.ndarray
        Array of potentials to split on different variables
        Doesn't need to sum to one

    Returns
    -------
    Split

    Examples
    --------
    >>> data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}), np.array([1, 1, 1]))
    >>> split = sample_split(data)
    >>> split.splitting_variable in data.variables
    True
    >>> split.splitting_value in data.X[split.splitting_variable]
    True
    """
    split_variable = np.random.choice(list(node.splittable_variables))
    split_value = node.data.random_splittable_value(split_variable)
    if split_value is None:
        return None
    return SplitCondition(split_variable, split_value)


