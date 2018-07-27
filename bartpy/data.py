from collections import namedtuple
from typing import Any, MutableMapping, Set

import pandas as pd
import numpy as np


class Split:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value

    def __str__(self):
        return self.splitting_variable + ": " + str(self.splitting_value)


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
        return {x for x in self.X.columns if len(set(self.X[x])) > 0}

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

    def random_variable(self) -> str:
        """
        Choose a variable at random from the set of splittable variables
        Returns
        -------
            str - a variable name that can be split on
        """
        return np.random.choice(np.array(list(self.variables)))[0][0]

    def random_value(self, variable: str) -> Any:
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
        >>> random_a = [data.random_value("a") for _ in range(100)]
        >>> np.all([x in [1, 2] for x in random_a])
        True
        >>> random_b = [data.random_value("b") for _ in range(100)]
        >>> np.all([x in [1] for x in random_b])
        True
        >>> unsplittable_data = Data(pd.DataFrame({"a": [1, 1], "b": [1, 1]}), np.array([1, 1, 1]))
        >>> unsplittable_data.random_value("a")
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
        print(-0.5 + (y - y_min) / (y_max - y_min))
        return pd.Series(-0.5 + (y - y_min) / (y_max - y_min))



def sample_split(data: Data, variable_prior=None) -> Split:
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
    split_variable = data.random_variable()
    split_value = data.random_value(split_variable)
    if split_value is None:
        return None
    return Split(split_variable, split_value)


if __name__ == "__main__":
    import doctest
    doctest.testmod()