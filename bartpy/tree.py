from abc import abstractmethod, ABC
from typing import List, Set, Generator, Optional

import numpy as np
import pandas as pd

from bartpy.data import Split, Data, sample_split_condition, SplitCondition
from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException


class TreeMutation(ABC):

    def __init__(self, kind: str, existing_node: 'TreeNode', updated_node: Optional['TreeNode']):
        if kind not in ["grow", "change", "prune"]:
            raise NotImplementedError("{} is not a supported proposal".format(kind))
        self.kind = kind
        self.existing_node = existing_node
        self.updated_node = updated_node

    def __str__(self):
        return "{} - {} => {}".format(self.kind, self.existing_node, self.updated_node)


class PruneMutation(TreeMutation):

    def __init__(self, existing_node: 'SplitNode', updated_node: 'LeafNode'):
        if not existing_node.is_leaf_parent():
            raise TypeError("Pruning only valid on leaf parents")
        super().__init__("prune", existing_node, updated_node)


class GrowMutation(TreeMutation):

    def __init__(self, existing_node: 'LeafNode', updated_node: 'SplitNode'):
        if not updated_node.is_leaf_parent():
            raise TypeError("Can only grow into Leaf parents")
        if not existing_node.is_leaf_node():
            raise TypeError("Can only grow Leaf nodes")
        super().__init__("grow", existing_node, updated_node)


class ChangeMutation(TreeMutation):

    def __init__(self, existing_node: 'SplitNode', updated_node: 'SplitNode'):
        if not existing_node.is_leaf_node():
            raise TypeError("Pruning only valid on leaf parents")
        super().__init__("change", existing_node, updated_node)


class TreeNode(ABC):

    def __init__(self, data: Data, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self._data = data
        self.depth = depth
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

    def is_leaf_parent(self) -> bool:
        return False

    @abstractmethod
    def predict(self) -> pd.Series:
        raise NotImplementedError()

    def is_split_node(self) -> bool:
        return False


class LeafNode(TreeNode):

    def __init__(self, data: Data, split: Split=None, depth=0):
        self._value = 0.0
        self._residuals = 0.0
        if split is None:
            self._split = Split([])
        else:
            self._split = split
        super().__init__(data, depth, None, None)

    def set_value(self, value: float) -> None:
        self._value = value
        # if isinstance(value, float):
        #     self._value = value
        # else:
        #     raise TypeError("LeafNode values can only be floats, found {}".format(type(value)))

    def residuals(self) -> np.ndarray:
        return self.data.y - self.current_value

    def update_node(self, mutation: TreeMutation) -> bool:
        return False

    def update_data(self, data: Data):
        self._data = data

    @property
    def current_value(self):
        return self._value

    def predict(self) -> pd.Series:
        return self.current_value

    def is_splittable(self) -> bool:
        return len(self.data.splittable_variables()) > 0

    @property
    def split(self) -> Split:
        return self._split

    def is_leaf_node(self):
        return True


class SplitNode(TreeNode):

    def __init__(self, data: Data, split: Split, left_child_node: LeafNode, right_child_node: LeafNode, depth=0):
        self.split = split
        super().__init__(data, depth, left_child_node, right_child_node)

    def is_leaf_parent(self) -> bool:
        return self.left_child.is_leaf_node() and self.right_child.is_leaf_node()

    def is_split_node(self) -> bool:
        return True

    def update_data(self, data: Data):
        self._data = data
        left_data = self.left_child.split.split_data(data)
        right_data = self.right_child.split.split_data(data)
        self.left_child.update_data(left_data)
        self.right_child.update_data(right_data)

    def predict(self) -> pd.Series:
        return pd.concat([self.left_child.predict(), self.right_child.predict()])

    def children_split(self) -> SplitCondition:
        return self.left_child.split.most_recent_split_condition()


class TreeStructure:
    """
    An encapsulation of the structure of the tree as a whole
    """

    def __init__(self, head: TreeNode):
        self.head = head
        self.cache_up_to_date = False
        self._prediction = np.zeros_like(self.head.data.y)
        head_downstream = list(self.head.downstream_generator())
        starting_leaves = [x for x in head_downstream if x.is_leaf_node()]
        self._leaf_nodes = starting_leaves
        starting_leaf_parents = [x for x in head_downstream if x.is_leaf_parent()]
        self._leaf_parents = starting_leaf_parents
        self._leaf_node_map = np.array([self.head] * self.head.data.n_obsv)
        self._split_nodes = [x for x in head_downstream if x.is_split_node()]

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
        return self._leaf_nodes + self._split_nodes

    def leaf_nodes(self) -> List[LeafNode]:
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
        return self._leaf_nodes

    def split_nodes(self) -> List[TreeNode]:
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
        return self._split_nodes

    def leaf_parents(self) -> List[SplitNode]:
        return [x for x in self.split_nodes() if x.is_leaf_parent()]

    def n_leaf_nodes(self) -> int:
        return len(self.leaf_nodes())

    def n_leaf_parents(self) -> int:
        return len(self.leaf_parents())

    def random_leaf_node(self) -> LeafNode:
        return np.random.choice(list(self.leaf_nodes()))

    def random_splittable_leaf_node(self) -> LeafNode:
        splittable_nodes = [x for x in self.leaf_nodes() if x.is_splittable()]
        if len(splittable_nodes) > 0:
            return np.random.choice(splittable_nodes)
        else:
            raise NoSplittableVariableException()

    def random_leaf_parent(self) -> SplitNode:
        leaf_parents = self.leaf_parents()
        if len(leaf_parents) == 0:
            raise NoPrunableNodeException
        return np.random.choice(leaf_parents)

    def update_node(self, mutation: TreeMutation) -> None:

        if self.head == mutation.existing_node:
            self.head = mutation.updated_node
        else:
            self.head.update_node(mutation)

        self.cache_up_to_date = False

        if mutation.kind == "prune":
            self._split_nodes.remove(mutation.existing_node)
            self._leaf_nodes.append(mutation.updated_node)
            self._leaf_nodes.remove(mutation.existing_node.left_child)
            self._leaf_nodes.remove(mutation.existing_node.right_child)

        if mutation.kind == "grow":
            self._leaf_nodes.remove(mutation.existing_node)
            self._leaf_nodes.append(mutation.updated_node.left_child)
            self._leaf_nodes.append(mutation.updated_node.right_child)
            self._split_nodes.append(mutation.updated_node)

        if mutation.kind == "change":
            self._leaf_nodes.remove(mutation.existing_node.left_child)
            self._leaf_nodes.remove(mutation.existing_node.right_child)
            self._leaf_nodes.append(mutation.updated_node.left_child)
            self._leaf_nodes.append(mutation.updated_node.right_child)
            self._split_nodes.remove(mutation.existing_node)
            self._split_nodes.append(mutation.updated_node)

    def predict(self) -> np.ndarray:
        if self.cache_up_to_date:
            return self._prediction
        for leaf in self.leaf_nodes():
            condition = leaf.split.condition(self.head.data)
            self._prediction[condition] = leaf.predict()
        return self._prediction

    def update_data(self, data: Data) -> None:
        self.cache_up_to_date = False
        for node in self.nodes():
            node.data._y = data.y[node.split.condition(data)]


def split_node(node: LeafNode, split_condition: SplitCondition) -> SplitNode:
    left_split = node.split + split_condition.left
    right_split = node.split + split_condition.right
    left_data = left_split.split_data(node.data)
    right_data = right_split.split_data(node.data)
    return SplitNode(node.data, node.split, LeafNode(left_data, left_split, depth=node.depth + 1), LeafNode(right_data, right_split, depth=node.depth+1), depth=node.depth)


def sample_split_node(node: LeafNode, variable_prior=None) -> SplitNode:
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
    >>> isinstance(new_node, SplitNode)
    True
    >>> len(new_node.left_child.data.X) + len(new_node.right_child.data.X)
    3

    >>> unsplittable_data = Data(pd.DataFrame({"a": [1, 1], "b": [1, 1]}), np.array([1, 1, 1]))
    >>> unsplittable_node = TreeNode(unsplittable_data)
    >>> sample_split_node(unsplittable_node) == unsplittable_node
    True
    """
    if not node.is_splittable():
        raise NoSplittableVariableException()
    else:
        condition = sample_split_condition(node.data, variable_prior)
        return split_node(node, condition)


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
    doctest.testmod()