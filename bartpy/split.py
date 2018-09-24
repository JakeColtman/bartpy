from copy import deepcopy
from typing import List, Optional, Tuple, Union
from operator import le, gt

from bartpy.data import Data

import pandas as pd
import numpy as np


class SplitCondition:
    """
    A representation of a split in feature space.
    The two main components are:

        - splitting_variable: which variable is being split on
        - splitting_value: the value being split on
                           all values less than or equal to this go left, all values greater go right

    """

    def __init__(self, splitting_variable: str, splitting_value: float, operator: Union[gt, le], condition=None):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = condition
        self.operator = operator

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def __eq__(self, other: 'SplitCondition'):
        return self.splitting_variable == other.splitting_variable and self.splitting_value == other.splitting_value and self.operator == other.operator

    def condition(self, data: Data, cached=True) -> np.ndarray:
        """
        Returns a Bool array indicating which side of the split each row of `Data` should go
        False => Left
        True => Right
        """
        if not cached or self._condition is None:
            self._condition = self.operator(data.X[self.splitting_variable].values, self.splitting_value)
        return self._condition

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
    """
    The Split class represents the conditioned data at any point in the decision tree
    It contains the logic for:

     - Maintaining a record of which rows of the covariate matrix are in the split
     - Being able to easily access a `Data` object with the relevant rows
     - Applying `SplitConditions` to further break up the data
    """

    def __init__(self, data: Data, split_conditions: List[SplitCondition]=None, combined_condition=None):
        if split_conditions is None:
            split_conditions = []
        self._conditions = split_conditions
        self._data = Data(data.X, deepcopy(data.y), cache=False)
        self._combined_condition = combined_condition
        self._conditioned_X = pd.DataFrame(self._data.X.values[self.condition()], columns=self._data.X.columns)
        self._conditioned_data = Data(self._conditioned_X, self._data._y[self.condition()])

    @property
    def data(self):
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
            return self.out_of_sample_condition(data)

    def out_of_sample_condition(self, X: pd.DataFrame):
        data = Data(X, np.array([0] * len(X)))
        condition = np.array([True] * len(X))
        for split_condition in self._conditions:
            condition = condition & split_condition.condition(data, cached=False)
        return condition

    def __add__(self, other: SplitCondition):
        return Split(self._data, self._conditions + [other], combined_condition=self.condition() & other.condition(self._data))

    def most_recent_split_condition(self) -> Optional[SplitCondition]:
        if len(self._conditions) > 0:
            return self._conditions[-1]
        else:
            return None

    def update_y(self, y):
        self._conditioned_data._y = y[self.condition()]
        self._data._y = y