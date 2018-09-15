from typing import List, Optional
from copy import deepcopy

from bartpy.data import Data

import pandas as pd
import numpy as np


def fancy_bool(x, bool_mask):
    return x[bool_mask]


class SplitCondition:
    """
    A representation of a split in feature space.
    The two main components are:
        - splitting_variable: which variable is being split on
        - splitting_value: the value being split on
                           all values less than or equal to this go left, all values greater go right
    """

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = None

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data: Data) -> np.ndarray:
        """
        Returns a Bool array indicating which side of the split each row of `Data` should go
        False => Left
        True => Right
        """
        if self._condition is None:
            self._condition = data.X[self.splitting_variable] > self.splitting_value
        return self._condition

    def left(self, data: Data) -> np.ndarray:
        """
        Returns a Bool array indicating whether each row should go into the left split.
        Inverse of self.right
        """
        return ~self.condition(data)

    def right(self, data: Data) -> np.ndarray:
        """
        Returns a Bool array indicating whether each row should go into the left split.
        Inverse of self.right
        """
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


def sample_split_condition(node) -> Optional[SplitCondition]:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws
        - draw a node to split on based on multinomial distribution
        - draw an observation within that variable to split on
    Returns None if there isn't a possible non-degenerate split
    """
    split_variable = np.random.choice(list(node.splittable_variables))
    split_value = node.data.random_splittable_value(split_variable)
    if split_value is None:
        return None
    return SplitCondition(split_variable, split_value)


