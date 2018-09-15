from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.split import Split, sample_split_condition, SplitCondition


class TreeNode(ABC):

    def __init__(self, split: Split, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self.depth = depth
        self._split = split
        self._left_child = left_child
        self._right_child = right_child

    def update_data(self, data: Data) -> None:
        self._split._data = data

    @property
    def data(self) -> Data:
        return self.split.data

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    def is_leaf_node(self) -> bool:
        return False

    def is_decision_node(self) -> bool:
        return False

    @property
    def split(self):
        return self._split

    @abstractmethod
    def update_y(self, y):
        raise NotImplementedError()


class LeafNode(TreeNode):

    def __init__(self, split: Split, depth=0):
        self._value = 0.0
        self._residuals = 0.0
        self._splittable_variables = None
        super().__init__(split, depth, None, None)

    @property
    def splittable_variables(self):
        if self._splittable_variables is None:
            self._splittable_variables = self.split.data.splittable_variables()
        return self._splittable_variables

    def set_value(self, value: float) -> None:
        self._value = value

    def residuals(self) -> np.ndarray:
        return self.data.y - self.predict()

    @property
    def current_value(self):
        return self._value

    def predict(self) -> float:
        return self.current_value

    def is_splittable(self) -> bool:
        return len(self.splittable_variables) > 0

    def is_leaf_node(self):
        return True

    def update_y(self, y):
        self.split.update_y(y)


class DecisionNode(TreeNode):

    def __init__(self, split: Split, left_child_node: Union[LeafNode, 'DecisionNode'], right_child_node: Union[LeafNode, 'DecisionNode'], depth=0):
        super().__init__(split, depth, left_child_node, right_child_node)

    def is_prunable(self) -> bool:
        return self.left_child.is_leaf_node() and self.right_child.is_leaf_node()

    def is_decision_node(self) -> bool:
        return True

    def variable_split_on(self) -> SplitCondition:
        return self.left_child.split.most_recent_split_condition()

    def update_y(self, y):
        self.split.update_y(y)
        self.left_child.update_y(y)
        self.right_child.update_y(y)


def split_node(node: LeafNode, split_condition: SplitCondition) -> DecisionNode:
    left_split, right_split = node.split + split_condition
    return DecisionNode(node.split,
                        LeafNode(left_split, depth=node.depth + 1),
                        LeafNode(right_split, depth=node.depth + 1),
                        depth=node.depth)


def sample_split_node(node: LeafNode, variable_prior=None) -> DecisionNode:
    """
    Split a leaf node into an internal node with two lead children
    The variable and value to split on is determined by sampling from their respective distributions

    Parameters
    ----------
    node - TreeNode
        The node to split
    variable_prior - np.ndarray
        Multinomial potentials to use as weights for selecting variable to split on
    Returns
    -------
        TreeNode
            New node with two leaf children

    Examples
    --------
    >>> data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}), np.array([1, 1, 1]))
    >>> node = TreeNode(data)
    >>> new_node = sample_split_node(node)
    >>> new_node.left_child is not None
    True
    >>> new_node.right_child is not None
    True
    >>> isinstance(new_node, DecisionNode)
    True
    >>> len(new_node.left_child.data.X) + len(new_node.right_child.data.X)
    3

    >>> unsplittable_data = Data(pd.DataFrame({"a": [1, 1], "b": [1, 1]}), np.array([1, 1, 1]))
    >>> unsplittable_node = TreeNode(unsplittable_data)
    >>> sample_split_node(unsplittable_node) == unsplittable_node
    True
    """
    condition = sample_split_condition(node, variable_prior)
    return split_node(node, condition)
