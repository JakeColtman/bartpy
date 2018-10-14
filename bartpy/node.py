from typing import Union, Tuple

import numpy as np

from bartpy.data import Data
from bartpy.split import Split, SplitCondition


class TreeNode:
    """
    A representation of a node in the Tree
    Contains two main types of information:
        - Data relevant for the node
        - Links to children nodes
    """
    def __init__(self, split: Split, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self.depth = depth
        self._split = split
        self._left_child = left_child
        self._right_child = right_child

    @property
    def data(self) -> Data:
        return self.split.data

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    @property
    def split(self):
        return self._split

    def update_y(self, y):
        self.split.update_y(y)
        if self.left_child is not None:
            self.left_child.update_y(y)
            self.right_child.update_y(y)


class LeafNode(TreeNode):
    """
    A representation of a leaf node in the tree
    In addition to the normal work of a `Node`, a `LeafNode` is responsible for:
        - Interacting with `Data`
        - Making predictions
    """

    def __init__(self, split: Split, depth=0, value=0.0):
        self._value = value
        super().__init__(split, depth, None, None)

    def set_value(self, value: float) -> None:
        self._value = value

    @property
    def current_value(self):
        return self._value

    def predict(self) -> float:
        return self.current_value

    def is_splittable(self) -> bool:
        return len(self.split.data.splittable_variables()) > 0


class DecisionNode(TreeNode):
    """
    A `DecisionNode` encapsulates internal node in the tree
    Unlike a `LeafNode`, it contains very little actual logic beyond tying the tree together
    """

    def __init__(self, split: Split, left_child_node: Union[LeafNode, 'DecisionNode'], right_child_node: Union[LeafNode, 'DecisionNode'], depth=0):
        super().__init__(split, depth, left_child_node, right_child_node)

    def is_prunable(self) -> bool:
        return type(self.left_child) == LeafNode and type(self.right_child) == LeafNode

    def variable_split_on(self) -> SplitCondition:
        return self.left_child.split.most_recent_split_condition()


def split_node(node: LeafNode, split_conditions: Tuple[SplitCondition, SplitCondition]) -> DecisionNode:
    """
    Converts a `LeafNode` into an internal `DecisionNode` by applying the split condition
    The left node contains all values for the splitting variable less than the splitting value
    """
    left_split = node.split + split_conditions[0]
    right_split = node.split + split_conditions[1]

    return DecisionNode(node.split,
                        LeafNode(left_split, depth=node.depth + 1),
                        LeafNode(right_split, depth=node.depth + 1),
                        depth=node.depth)


def deep_copy_node(node: TreeNode):
    if type(node) == LeafNode:
        return LeafNode(node.split.out_of_sample_conditioner(), value=node.current_value, depth=node.depth)
    elif type(node) == DecisionNode:
        return DecisionNode(node.split.out_of_sample_conditioner(), node.left_child, node.right_child, depth=node.depth)
    else:
        raise TypeError("Unsupported node type")
