from operator import le, gt
from typing import Callable, List

import numpy as np


class SplitCondition(object):
    """
    A representation of a split in feature space.
    The two main components are:

        - splitting_variable: which variable is being split on
        - splitting_value: the value being split on
                           all values less than or equal to this go left, all values greater go right

    """

    def __init__(self, 
                 splitting_variable: int, 
                 splitting_value: float, 
                 operator: Callable[[float, float], bool], 
                 condition=None):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._condition = condition
        self.operator = operator

    def __str__(self):
        return str(self.splitting_variable) + ": " + str(self.splitting_value)

    def __eq__(self, other: 'SplitCondition'):
        return self.splitting_variable == other.splitting_variable and self.splitting_value == other.splitting_value and self.operator == other.operator


class CombinedVariableCondition(object):

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


class CombinedCondition(object):

    def __init__(self, variables: List[int], conditions: List[SplitCondition]):
        self.variables = variables
        self._variable_conditions = None
        self.conditions = conditions
        if len(conditions) > 0:
            self.splitting_variable = conditions[-1].splitting_variable
        else:
            self.splitting_variable = None

    def condition(self, X: np.ndarray) -> np.ndarray:
        c = np.array([True] * len(X))
        for variable in self.variables:
            c = c & (X[:, variable] > self.variable_conditions[variable].min_value) & (X[:, variable] <= self.variable_conditions[variable].max_value)
        return c

    def __add__(self, other: SplitCondition):
        return CombinedCondition(list(self.variables), self.conditions + [other])

    def most_recent_split_condition(self):
        if len(self.conditions) == 0:
            return None
        else:
            return self.conditions[-1]

    @property
    def variable_conditions(self):
        if self._variable_conditions is None:
            self._variable_conditions = {v: CombinedVariableCondition(v, -np.inf, np.inf) for v in self.variables}
            for condition in self.conditions:
                self._variable_conditions[condition.splitting_variable] = self._variable_conditions[condition.splitting_variable].add_condition(
                    condition)
        return self._variable_conditions