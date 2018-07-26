import numpy as np
import pandas as pd

from bartpy.data import Split, Data, sample_split


class TreeStructure:

    def __init__(self):
        pass

    def predict(self, X):
        pass


class TreeNode:

    def __init__(self, data: Data, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self._data = data
        self._left_child = left_child
        self._right_child = right_child

    @property
    def data(self) -> Data:
        return self._data

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    def update_left_child(self, node: 'TreeNode'):
        self._left_child = node

    def update_right_child(self, node: 'TreeNode'):
        self._right_child = node


class SplitNode(TreeNode):

    def __init__(self, data: Data, split: Split, left_child_node: TreeNode, right_child_node: TreeNode):
        self.split = split
        super().__init__(data, left_child_node, right_child_node)


def split_node(node: TreeNode, variable_prior=None) -> TreeNode:
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
    >>> data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}))
    >>> node = TreeNode(data)
    >>> new_node = split_node(node)
    >>> new_node.left_child is not None
    True
    >>> new_node.right_child is not None
    True
    >>> isinstance(new_node, SplitNode)
    True
    >>> len(new_node.left_child.data.data) + len(new_node.right_child.data.data)
    3
    """
    split = sample_split(node.data, variable_prior)
    split_data = node.data.split_data(split)
    left_child_node = TreeNode(split_data.left_data)
    right_child_node = TreeNode(split_data.right_data)

    return SplitNode(node.data, split, left_child_node, right_child_node)


def is_terminal(depth: int, alpha: float, beta: float) -> bool:
    """
    Determine whether a node is a leaf node or should be split on

    Parameters
    ----------
    depth
    alpha
    beta

    Returns
    -------
    bool
        True means no more splits should be done
    """
    r = np.random.uniform(0, 1)
    return r < alpha * np.power(1 + depth, beta)


def sample_tree_structure_from_node(node: TreeNode, depth: int, alpha: float, beta: float, variable_prior=None):
    terminal = is_terminal(depth, alpha, beta)
    if terminal:
        return node
    else:
        updated_node = split_node(node, variable_prior)
        updated_node.update_left_child(sample_tree_structure_from_node(split_node.left_child, depth + 1, alpha, beta, variable_prior))
        updated_node.update_right_child(sample_tree_structure_from_node(split_node.right_child, depth + 1, alpha, beta, variable_prior))
        return updated_node


def sample_tree_structure(data: Data, alpha: float = 0.95, beta: float = 2, variable_prior=None):
    head = TreeNode(data)
    tree = sample_tree_structure_from_node(head, 0, alpha, beta, variable_prior)
    return tree


if __name__ == "__main__":
    data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}))
    node = TreeNode(data)
    new_node = split_node(node)
    print(new_node.left_child.data.data)
    print(new_node.right_child.data.data)
