from copy import deepcopy
from typing import List, Optional, Union
from operator import le, gt

from bartpy.data import Data

import numpy as np


class SplitCondition:
    """
    A representation of a split in feature space.
    The two main components are:

        - splitting_variable: which variable is being split on
        - splitting_value: the value being split on
                           all values less than or equal to this go left, all values greater go right

    """

    def __init__(self, splitting_variable: int, splitting_value: float, operator: Union[gt, le], condition=None):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = condition
        self.operator = operator

    def __str__(self):
        return str(self.splitting_variable) + ": " + str(self.splitting_value)

    def __eq__(self, other: 'SplitCondition'):
        return self.splitting_variable == other.splitting_variable and self.splitting_value == other.splitting_value and self.operator == other.operator

    def condition(self, data: Data, cached=True) -> np.ndarray:
        """
        Returns a Bool array indicating whether each row falls into this side of the split condition
        """
        if not cached or self._condition is None:
            self._condition = self.operator(data.X[:, self.splitting_variable], self.splitting_value)
        return self._condition


class CombinedVariableCondition:

    def __init__(self, splitting_variable: int, min_value: float, max_value: float):
        self.splitting_variable = splitting_variable
        self.min_value, self.max_value = min_value, max_value

    def add_condition(self, split_condition: SplitCondition) -> 'CombinedVariableCondition':
        if self.splitting_variable != split_condition.splitting_variable:
            return self
        if split_condition.operator == gt and split_condition.splitting_value > self.min_value:
            return CombinedVariableCondition(self.splitting_variable, split_condition.splitting_value, self.max_value)
        elif split_condition.operator == le and split_condition.splitting_value < self.max_value:
            return CombinedVariableCondition(self.splitting_variable, self.min_value, split_condition.splitting_value)
        else:
            return self


class CombinedCondition:

    def __init__(self, variables: List[int], conditions: List[SplitCondition]):
        self.variables = {v: CombinedVariableCondition(v, -np.inf, np.inf) for v in variables}
        for condition in conditions:
            self.variables[condition.splitting_variable] = self.variables[condition.splitting_variable].add_condition(condition)
        if len(conditions) > 0:
            self.splitting_variable = conditions[-1].splitting_variable
        else:
            self.splitting_variable = None

    def condition(self, X: np.ndarray):
        c = np.array([True] * len(X))
        for variable in self.variables.keys():
            c = c & (X[:, variable] > self.variables[variable].min_value) & (X[:, variable] <= self.variables[variable].max_value)
        return c


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
        self._data = Data(data.X, deepcopy(data.y), cache=False, unique_columns=data.unique_columns)
        self._conditions = split_conditions
        self._combined_condition = combined_condition
        self._conditioned_X = self._data.X[self.condition()]
        self._conditioned_data = Data(self._conditioned_X, self._data._y[self.condition()], unique_columns=data.unique_columns)
        self._combined_conditioner = None

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

    def out_of_sample_condition(self, X: np.ndarray):
        data = Data(X, np.array([0] * len(X)), cache=False)
        return self.out_of_sample_conditioner().condition(X)

    def out_of_sample_conditioner(self) -> CombinedCondition:
        if self._combined_conditioner is None:
            self._combined_conditioner = CombinedCondition(self.data.variables, self._conditions)
        return self._combined_conditioner

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