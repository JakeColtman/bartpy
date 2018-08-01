from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, MutableMapping, Set, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from bartpy.errors import NoSplittableVariableException


class SplitCondition(ABC):

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    @property
    def left(self):
        return LTESplitCondition(self.splitting_variable, self.splitting_value)

    @property
    def right(self):
        return GTSplitCondition(self.splitting_variable, self.splitting_value)


class GTSplitCondition:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data: 'Data'):
        return data.X[self.splitting_variable] > self.splitting_value


class LTESplitCondition:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)

    def condition(self, data: 'Data'):
        return data.X[self.splitting_variable] <= self.splitting_value


class Split:

    def __init__(self, data: 'Data', split_conditions: List[Union[LTESplitCondition, GTSplitCondition]]):
        self._conditions = split_conditions
        self._data = data
        self._combined_condition = self.combined_condition(self._data)

    def combined_condition(self, data):
        if len(self._conditions) == 0:
            return [True] * data.n_obsv
        if len(self._conditions) == 1:
            return self._conditions[0].condition(data)
        else:
            final_condition = self._conditions[0].condition(data)
            for c in self._conditions[1:]:
                final_condition = final_condition & c.condition(data)
            return final_condition

    def condition(self, data: 'Data'=None):
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

    def split_data(self, data: 'Data') -> 'Data':
        data = Data(data.X[self.condition(data)], data.y[self.condition(data)])
        return data


SplitData = namedtuple("SplitData", ["left_data", "right_data"])


class Data:
    """
    Encapsulates feature data
    Useful for providing cached access to commonly used functions of the data

    Examples
    --------
    >>> data_pd = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]})
    >>> data = Data(data_pd, np.array([1, 1, 1]))
    >>> data.variables == {"a", "b"}
    True
    >>> data.unique_values("a")
    {1, 2, 3}
    >>> data.unique_values("b")
    {1, 2}
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, normalize=False):
        self._unique_values_cache: MutableMapping[str, Set[Any]] = {}
        self._X = X
        if normalize:
            self._y = self.normalize_y(y)
        else:
            self._y = y

    @property
    def y(self) -> pd.Series:
        return self._y

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    def splittable_variables(self) -> Set[str]:
        return {x for x in self.X.columns if len(set(self.X[x])) > 1}

    @property
    def variables(self) -> Set[str]:
        """
        The set of variable names the data contains.
        Of dimensionality p

        Returns
        -------
        Set[str]
        """
        return set(self.X.columns)

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
        return np.random.choice(np.array(list(splittable_variables)), 1)[0][0]

    def random_splittable_value(self, variable: str) -> Any:
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

        Examples
        --------
        >>> data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}), np.array([1, 1, 1]))
        >>> random_a = [data.random_splittable_value("a") for _ in range(100)]
        >>> np.all([x in [1, 2] for x in random_a])
        True
        >>> random_b = [data.random_splittable_value("b") for _ in range(100)]
        >>> np.all([x in [1] for x in random_b])
        True
        >>> unsplittable_data = Data(pd.DataFrame({"a": [1, 1], "b": [1, 1]}), np.array([1, 1, 1]))
        >>> unsplittable_data.random_splittable_value("a")
        """
        possible_values = self.unique_values(variable)
        possible_values = possible_values - {np.max(list(possible_values))}
        if len(possible_values) == 0:
            return None
        return np.random.choice(np.array(list(possible_values)))

    def unique_values(self, variable: str) -> Set[Any]:
        """
        Set of all values a variable takes in the feature set

        Parameters
        ----------
        variable - str
            name of the variable

        Returns
        -------
        Set[Any] - all possible values
        """
        if variable not in self._unique_values_cache:
            self._unique_values_cache[variable] = set(self.X[variable])
        return self._unique_values_cache[variable]

    def split_data(self, split):
        lhs_condition = self.X[split.splitting_variable] <= split.splitting_value
        rhs_condition = self.X[split.splitting_variable] > split.splitting_value

        lhs = Data(self.X[lhs_condition], self.y[lhs_condition])
        rhs = Data(self.X[rhs_condition], self.y[rhs_condition])

        return SplitData(lhs, rhs)

    @property
    def n_obsv(self) -> int:
        return len(self.X)

    @property
    def n_splittable_variables(self) -> int:
        return len(self.splittable_variables())

    def n_unique_values(self, variable: str) -> int:
        return len(self.unique_values(variable))

    @staticmethod
    def normalize_y(y: pd.Series) -> pd.Series:
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
        return pd.Series(-0.5 + (y - y_min) / (y_max - y_min))


def sample_split_condition(data: Data, variable_prior=None) -> Optional[SplitCondition]:
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
    split_variable = data.random_splittable_variable()
    split_value = data.random_splittable_value(split_variable)
    if split_value is None:
        return None
    return SplitCondition(split_variable, split_value)


if __name__ == "__main__":
    import doctest
    doctest.testmod()