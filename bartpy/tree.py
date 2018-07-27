from abc import abstractclassmethod
from typing import List, Set, Generator, Optional

import numpy as np
import pandas as pd

from bartpy.data import Split, Data, sample_split


class TreeMutation:

    def __init__(self, kind: str, existing_node: 'TreeNode', updated_node: Optional['TreeNode']):
        if kind not in ["grow", "change", "prune"]:
            raise NotImplementedError("{} is not a supported proposal".format(kind))
        self.kind = kind
        self.existing_node = existing_node
        self.updated_node = updated_node


class TreeNode:

    def __init__(self, data: Data, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self._data = data
        self._left_child = left_child
        self._right_child = right_child

    def update_node(self, mutation: TreeMutation) -> bool:
        if self.left_child == mutation.existing_node:
            self._left_child = mutation.updated_node
            return True
        elif self.right_child == mutation.existing_node:
            self._right_child = mutation.updated_node
            return True
        else:
            found_left = self.left_child.update_node(mutation)
            if found_left:
                return True
            else:
                found_right = self.right_child.update_node(mutation)
                return found_right

    def downstream_generator(self) -> Generator['TreeNode', None, None]:
        """
        An generator of all descendants of a node
        i.e. the children of the node and their children and so on
        Returns
        -------
            Generator['TreeNode', None, None]

        Examples
        --------
        >>> a, b, c, = TreeNode(None), TreeNode(None), TreeNode(None)
        >>> a.update_left_child(b)
        >>> b.update_left_child(c)
        >>> nodes = list(a.downstream_generator())
        >>> np.all([x[0] == x[1] for x in zip([c, b, a], nodes)])
        True
        >>> nodes = list(b.downstream_generator())
        >>> np.all([x[0] == x[1] for x in zip([c, b], nodes)])
        True
        """
        if self.left_child is not None:
            for x in self.left_child.downstream_generator():
                yield x
        if self.right_child is not None:
            for x in self.right_child.downstream_generator():
                yield x
        yield self

    def residuals(self) -> np.ndarray:
        return np.zeros_like(self.data.y)

    def update_data(self, data: Data) -> None:
        raise NotImplementedError()

    def downstream_residuals(self):
        return self.residuals() + self.left_child.residuals() + self.right_child.residuals()

    @property
    def data(self) -> Data:
        return self._data

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    def update_left_child(self, node: Optional['TreeNode']):
        self._left_child = node

    def update_right_child(self, node: Optional['TreeNode']):
        self._right_child = node

    def is_leaf_node(self) -> bool:
        return self.left_child is None and self.right_child is None

    @abstractclassmethod
    def predict(self, data: Data) -> pd.Series:
        raise NotImplementedError()


class SplitNode(TreeNode):

    def __init__(self, data: Data, split: Split, left_child_node: TreeNode, right_child_node: TreeNode):
        self.split = split
        super().__init__(data, left_child_node, right_child_node)

    def update_data(self, data: Data):
        left_data, right_data = data.split_data(self.split)
        self.left_child.update_data(left_data)
        self.right_child.update_data(right_data)

    def predict(self, data: Data) -> pd.Series:
        left_predict_data, right_predict_data = data.split_data(self.split)
        return pd.concat([self.left_child.predict(left_predict_data), self.right_child.predict(right_predict_data)])


class LeafNode(TreeNode):

    def __init__(self, data):
        self._value = 0.0
        self._residuals = 0.0
        super().__init__(data, None, None)

    def set_value(self, value: float) -> None:
        self._value = value

    def residuals(self) -> np.ndarray:
        return self.data.y - self.current_value

    def update_data(self, data: Data):
        self._data = data

    @property
    def current_value(self):
        return self._value

    def predict(self, data: Data) -> pd.Series:
        y = data.y.reset_index()
        y["prediction"] = self.current_value
        y = y.set_index("index")
        print(y["prediction"])
        return y["prediction"]


class TreeStructure:
    """
    An encapsulation of the structure of the tree as a whole
    """

    def __init__(self, head: TreeNode):
        self.head = head

    def nodes(self) -> List[TreeNode]:
        """

        Returns
        -------
            List[TreeNode]

        Examples
        --------
        >>> a, b, c, = TreeNode(None), TreeNode(None), TreeNode(None)
        >>> a.update_left_child(b)
        >>> b.update_left_child(c)
        >>> structure = TreeStructure(a)
        >>> nodes = structure.nodes()
        >>> len(nodes)
        3
        >>> a in nodes
        True
        >>> 1 in nodes
        False
        """
        all_nodes = []
        for n in self.head.downstream_generator():
            all_nodes.append(n)
        return all_nodes

    def leaf_nodes(self) -> Set[LeafNode]:
        """

        Returns
        -------
            List[TreeNode]

        Examples
        --------
        >>> a, b, c, = TreeNode(None), TreeNode(None), TreeNode(None)
        >>> a.update_left_child(b)
        >>> b.update_left_child(c)
        >>> structure = TreeStructure(a)
        >>> nodes = structure.leaf_nodes()
        >>> len(nodes)
        1
        >>> c == list(nodes)[0]
        True
        """
        return {x for x in self.nodes() if x.is_leaf_node()}

    def split_nodes(self) -> Set[TreeNode]:
        """

        Returns
        -------
            List[TreeNode]

        Examples
        --------
        >>> a, b, c, = TreeNode(None), TreeNode(None), TreeNode(None)
        >>> a.update_left_child(b)
        >>> a.update_right_child(c)
        >>> structure = TreeStructure(a)
        >>> nodes = structure.split_nodes()
        >>> len(nodes)
        1
        >>> a == list(nodes)[0]
        True
        """
        return {x for x in self.nodes() if not x.is_leaf_node()}

    def leaf_parents(self) -> Set[SplitNode]:
        split_nodes = self.split_nodes()
        leaf_parents = {x for x in split_nodes if x.left_child.is_leaf_node() and x.right_child.is_leaf_node()}
        return leaf_parents

    def random_leaf_node(self) -> LeafNode:
        return np.random.choice(list(self.leaf_nodes()))

    def random_leaf_parent(self) -> SplitNode:
        return np.random.choice(list(self.leaf_parents()))

    def residuals(self) -> np.ndarray:
        return self.head.downstream_residuals()

    def update_node(self, mutation: TreeMutation):
        self.head.update_node(mutation)

    def predict(self, data: Data) -> pd.Series:
        return self.head.predict(data).sort_index()

    def update_data(self, data: Data) -> None:
        return self.head.update_data(data)

def split_node(node: LeafNode, variable_prior=None) -> SplitNode:
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
    >>> new_node = split_node(node)
    >>> new_node.left_child is not None
    True
    >>> new_node.right_child is not None
    True
    >>> isinstance(new_node, SplitNode)
    True
    >>> len(new_node.left_child.data.X) + len(new_node.right_child.data.X)
    3

    >>> unsplittable_data = Data(pd.DataFrame({"a": [1, 1], "b": [1, 1]}), np.array([1, 1, 1]))
    >>> unsplittable_node = TreeNode(unsplittable_data)
    >>> split_node(unsplittable_node) == unsplittable_node
    True
    """
    split = sample_split(node.data, variable_prior)
    if split is None:
        return node
    split_data = node.data.split_data(split)
    left_child_node = LeafNode(split_data.left_data)
    right_child_node = LeafNode(split_data.right_data)

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
    return r < alpha * np.power(1 + depth, -beta)


def sample_tree_structure_from_node(node: TreeNode, depth: int, alpha: float, beta: float, variable_prior=None) -> TreeNode:
    if depth == 0:
        updated_node = split_node(node)
        updated_node.update_left_child(sample_tree_structure_from_node(updated_node.left_child, depth + 1, alpha, beta, variable_prior))
        updated_node.update_right_child(sample_tree_structure_from_node(updated_node.right_child, depth + 1, alpha, beta, variable_prior))
        return updated_node

    terminal = is_terminal(depth, alpha, beta)
    if terminal:
        return node
    else:
        updated_node = split_node(node, variable_prior)
        if updated_node == node:
            return updated_node
        updated_node.update_left_child(sample_tree_structure_from_node(updated_node.left_child, depth + 1, alpha, beta, variable_prior))
        updated_node.update_right_child(sample_tree_structure_from_node(updated_node.right_child, depth + 1, alpha, beta, variable_prior))
        return updated_node


def sample_tree_structure(data: Data, alpha: float, beta: float, variable_prior=None) -> TreeStructure:
    node = TreeNode(data)
    head = sample_tree_structure_from_node(node, 0, alpha, beta, variable_prior)
    return TreeStructure(head)



#
# if __name__ == "__main__":
#     data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]}))
#     node = TreeNode(data)
#     new_node = split_node(node)
#     print(new_node.left_child.data.data)
#     print(new_node.right_child.data.data)
#
#     tree_structure = sample_tree_structure(data, 0.5)
#     head = tree_structure.head
#     print(head.split)
#     print(head.left_child.data.data)
#     print(head.right_child.data.data)

if __name__ == "__main__":
    import doctest
    doctest.testmod()#extraglobs={'t': TreeNode()})