from abc import abstractmethod, ABC
from typing import List, Set, Generator, Optional, Union

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.split import Split, sample_split_condition, SplitCondition, LTESplitCondition, GTSplitCondition


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


class LeafNode(TreeNode):

    def __init__(self, split: Split, depth=0):
        self._value = 0.0
        self._residuals = 0.0
        self.splittable_variables = split.data.splittable_variables()
        super().__init__(split, depth, None, None)

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


class DecisionNode(TreeNode):

    def __init__(self, split: Split, left_child_node: LeafNode, right_child_node: LeafNode, depth=0):
        super().__init__(split, depth, left_child_node, right_child_node)

    def is_prunable(self) -> bool:
        return self.left_child.is_leaf_node() and self.right_child.is_leaf_node()

    def is_decision_node(self) -> bool:
        return True

    def variable_split_on(self) -> Union[LTESplitCondition, GTSplitCondition]:
        return self.left_child.split.most_recent_split_condition()


def split_node(node: LeafNode, split_condition: SplitCondition) -> DecisionNode:
    return DecisionNode(node.split,
                        LeafNode(node.split + split_condition.left, depth=node.depth + 1),
                        LeafNode(node.split + split_condition.right, depth=node.depth + 1),
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
