from typing import List, Optional ,Tuple
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

    def __eq__(self, other: 'SplitCondition'):
        return self.splitting_variable == other.splitting_variable and self.splitting_value == other.splitting_value

    def condition(self, data: Data, cached=True) -> np.ndarray:
        """
        Returns a Bool array indicating which side of the split each row of `Data` should go
        False => Left
        True => Right
        """
        if not cached or self._condition is None:
            self._condition = data.X[self.splitting_variable] > self.splitting_value
        return self._condition

    def left_condition(self, data: Data, cached=True):
        return ~self.condition(data, cached)

    def left(self, data: Data) -> Tuple['SplitCondition', np.ndarray]:
        """
        Returns a Bool array indicating whether each row should go into the left split.
        Inverse of self.right
        """
        left_self = deepcopy(self)
        left_self.condition = self.left_condition
        return left_self, self.left_condition(data)

    def right(self, data: Data) -> Tuple['SplitCondition', np.ndarray]:
        """
        Returns a Bool array indicating whether each row should go into the left split.
        Inverse of self.right
        """
        return self, self.condition(data)


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

    def out_of_sample_condition(self, X: pd.DataFrame):
        data = Data(X, np.array([0] * len(X)))
        condition = np.array([True] * len(X))
        for split_condition in self._conditions:
            condition = condition & split_condition.condition(data, cached=False)
        return condition

    def __add__(self, other: SplitCondition):
        left_split, left_condition = other.left(self._data)
        right_split, right_condition = other.right(self._data)
        left = Split(self._data, self._conditions + [left_split], combined_condition=self.condition() & left_condition)
        right = Split(self._data, self._conditions + [right_split], combined_condition=self.condition() & right_condition)
        return left, right

    def most_recent_split_condition(self) -> Optional[SplitCondition]:
        if len(self._conditions) > 0:
            return self._conditions[-1]
        else:
            return None

    def update_y(self, y):
        self._cache_up_to_date = False
        self._data._y = y
