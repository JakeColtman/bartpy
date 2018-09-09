from abc import abstractmethod, ABC
from typing import List, Set, Generator, Optional, Union

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.split import Split, sample_split_condition, SplitCondition, LTESplitCondition, GTSplitCondition


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
        if not existing_node.is_leaf_parent():
            raise TypeError("Pruning only valid on leaf parents")
        super().__init__("change", existing_node, updated_node)


class TreeNode(ABC):

    def __init__(self, split: Split, depth: int, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self.depth = depth
        self._split = split
        self._left_child = left_child
        self._right_child = right_child

    def mutate(self, mutation: TreeMutation) -> bool:
        if self.left_child == mutation.existing_node:
            self._left_child = mutation.updated_node
            return True
        elif self.right_child == mutation.existing_node:
            self._right_child = mutation.updated_node
            return True
        else:
            found_left = self.left_child.mutate(mutation)
            if found_left:
                return True
            else:
                found_right = self.right_child.mutate(mutation)
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
        return self.left_child is None and self.right_child is None

    def is_leaf_parent(self) -> bool:
        return False

    def is_split_node(self) -> bool:
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
        return self.data.y - self.current_value

    def mutate(self, mutation: TreeMutation) -> bool:
        return False

    @property
    def current_value(self):
        return self._value

    def predict(self) -> float:
        return self.current_value

    def is_splittable(self) -> bool:
        return len(self.splittable_variables) > 0

    def is_leaf_node(self):
        return True


class SplitNode(TreeNode):

    def __init__(self, split: Split, left_child_node: LeafNode, right_child_node: LeafNode, depth=0):
        super().__init__(split, depth, left_child_node, right_child_node)

    def is_leaf_parent(self) -> bool:
        return self.left_child.is_leaf_node() and self.right_child.is_leaf_node()

    def is_split_node(self) -> bool:
        return True

    def split_on(self) -> Union[LTESplitCondition, GTSplitCondition]:
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
        self._leaf_nodes = [x for x in head_downstream if x.is_leaf_node()]
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
        return self._split_nodes + self._leaf_nodes

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

    def splittable_leaf_nodes(self):
        return [x for x in self.leaf_nodes() if x.is_splittable()]

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
        return [x for x in self._split_nodes if x.is_leaf_parent()]

    def n_leaf_nodes(self) -> int:
        return len(self.leaf_nodes())

    def n_leaf_parents(self) -> int:
        return len(self.leaf_parents())

    def random_leaf_node(self) -> LeafNode:
        return np.random.choice(self.leaf_nodes())

    def random_splittable_leaf_node(self) -> LeafNode:
        splittable_nodes = self.splittable_leaf_nodes()
        return splittable_nodes[0]
        if len(splittable_nodes) > 0:
            return np.random.choice(splittable_nodes)
        else:
            raise NoSplittableVariableException()

    def random_leaf_parent(self) -> SplitNode:
        leaf_parents = self.leaf_parents()
        if len(leaf_parents) == 0:
            raise NoPrunableNodeException
        return np.random.choice(leaf_parents)

    def mutate(self, mutation: TreeMutation) -> None:
        self.cache_up_to_date = False

        if self.head == mutation.existing_node:
            self.head = mutation.updated_node
        else:
            self.head.mutate(mutation)

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

    def update_y(self, y: np.ndarray) -> None:
        self.cache_up_to_date = False
        for node in self.nodes():
            node._split._data._y = y

    def predict(self) -> np.ndarray:
        if self.cache_up_to_date:
            return self._prediction

        for leaf in self.leaf_nodes():
            self._prediction[leaf.split.condition()] = leaf.predict()
        self.cache_up_to_date = True
        return self._prediction


def split_node(node: LeafNode, split_condition: SplitCondition) -> SplitNode:
    return SplitNode(node.split,
                     LeafNode(node.split + split_condition.left, depth=node.depth + 1),
                     LeafNode(node.split + split_condition.right, depth=node.depth+1),
                     depth=node.depth)


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
    condition = sample_split_condition(node, variable_prior)
    return split_node(node, condition)


if __name__ == "__main__":
    import doctest
    doctest.testmod()