from collections import namedtuple
from typing import Any, MutableMapping, Set, Optional

import pandas as pd
import numpy as np

from bartpy.errors import NoSplittableVariableException


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
            self.original_y_min, self.original_y_max = y.min(), y.max()
            self._y = self.normalize_y(y)
        else:
            self._y = y

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    def splittable_variables(self) -> Set[str]:
        return {x for x in self.X.columns if self.n_unique_values(x) > 1}

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
        possible_values = possible_values[possible_values != np.max(possible_values)]
        if len(possible_values) == 0:
            return None
        return np.random.choice(possible_values)

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
        return self.X[variable]

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

    def unnormalize_y(self, y: np.ndarray) -> np.ndarray:
        distance_from_min = y - (-0.5)
        total_distance = (self.original_y_max - self.original_y_min)
        return self.original_y_min + (distance_from_min * total_distance)

    @property
    def unnormalized_y(self) -> np.ndarray:
        return self.unnormalize_y(self.y)


if __name__ == "__main__":
    import doctest
    doctest.testmod()