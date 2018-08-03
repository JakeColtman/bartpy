from abc import ABC
from typing import List, Optional, Union
from copy import deepcopy

from bartpy.data import Data

import numpy as np


class SplitCondition(ABC):

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value
        self._left, self._right = None, None

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    @property
    def left(self):
        if self._left is None:
            self._left = LTESplitCondition(self.splitting_variable, self.splitting_value)
        return self._left

    @property
    def right(self):
        if self._right is None:
            self._right = GTSplitCondition(self.splitting_variable, self.splitting_value)
        return self._right


class GTSplitCondition:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data: Data):
        return data.X[self.splitting_variable] > self.splitting_value


class LTESplitCondition:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data: Data):
        return data.X[self.splitting_variable] <= self.splitting_value


class Split:

    def __init__(self, data: Data, split_conditions: List[Union[LTESplitCondition, GTSplitCondition]]):
        self._conditions = split_conditions
        self._data = deepcopy(data)
        self._combined_condition = self.combined_condition(self._data)

    @property
    def data(self):
        return Data(self._data.X[self.condition()], self._data.y[self.condition()])

    def combined_condition(self, data):
        if len(self._conditions) == 0:
            return np.array([True] * data.n_obsv)
        if len(self._conditions) == 1:
            return self._conditions[0].condition(data)
        else:
            final_condition = self._conditions[0].condition(data)
            for c in self._conditions[1:]:
                final_condition = final_condition & c.condition(data)
            return final_condition

    def condition(self, data: Data=None):
        if data is None:
            return self._combined_condition
        else:
            return self.combined_condition(data)

    def __add__(self, other: Union[LTESplitCondition, GTSplitCondition]):
        return Split(self._data, self._conditions + [other])

    def most_recent_split_condition(self) -> Optional[Union[LTESplitCondition, GTSplitCondition]]:
        if len(self._conditions) > 0:
            return self._conditions[-1]
        else:
            return None


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


