from typing import List

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation
from bartpy.node import TreeNode, LeafNode, DecisionNode


class Tree:
    """
    An encapsulation of the structure of the tree as a whole
    """

    def __init__(self, nodes: List[TreeNode]):
        self._nodes = nodes
        self.cache_up_to_date = False
        self._prediction = np.zeros_like(self._nodes[0]._split._data._y)

    @property
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
        >>> structure = Tree(a)
        >>> nodes = structure.nodes()
        >>> len(nodes)
        3
        >>> a in nodes
        True
        >>> 1 in nodes
        False
        """
        return self._nodes

    @property
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
        >>> structure = Tree(a)
        >>> nodes = structure.leaf_nodes()
        >>> len(nodes)
        1
        >>> c == list(nodes)[0]
        True
        """
        return [x for x in self._nodes if x.is_leaf_node()]

    @property
    def splittable_leaf_nodes(self):
        return [x for x in self.leaf_nodes if x.is_splittable()]

    @property
    def decision_nodes(self) -> List[DecisionNode]:
        """

        Returns
        -------
            List[TreeNode]

        Examples
        --------
        >>> a, b, c, = TreeNode(None), TreeNode(None), TreeNode(None)
        >>> a.update_left_child(b)
        >>> a.update_right_child(c)
        >>> structure = Tree(a)
        >>> nodes = structure.split_nodes()
        >>> len(nodes)
        1
        >>> a == list(nodes)[0]
        True
        """
        return [x for x in self._nodes if x.is_decision_node()]

    @property
    def prunable_decision_nodes(self) -> List[DecisionNode]:
        return [x for x in self.decision_nodes if x.is_prunable()]

    def update_y(self, y: np.ndarray) -> None:
        self.cache_up_to_date = False
        for node in self.nodes:
            node.split.update_y(y)

    def predict(self) -> np.ndarray:
        if self.cache_up_to_date:
            return self._prediction
        for leaf in self.leaf_nodes:
            self._prediction[leaf.split.condition()] = leaf.predict()
        self.cache_up_to_date = True
        return self._prediction


def random_splittable_leaf_node(tree: Tree) -> LeafNode:
    splittable_nodes = tree.splittable_leaf_nodes
    if len(splittable_nodes) > 0:
        return np.random.choice(splittable_nodes)
    else:
        raise NoSplittableVariableException()


def random_prunable_decision_node(tree: Tree) -> DecisionNode:
    leaf_parents = tree.prunable_decision_nodes
    if len(leaf_parents) == 0:
        raise NoPrunableNodeException
    return np.random.choice(leaf_parents)


def n_prunable_decision_nodes(tree: Tree) -> int:
    return len(tree.prunable_decision_nodes)


def n_splittable_leaf_nodes(tree: Tree) -> int:
    return len(tree.splittable_leaf_nodes)


def mutate(tree: Tree, mutation: TreeMutation) -> None:

    tree.cache_up_to_date = False

    if mutation.kind == "prune":
        tree._nodes.remove(mutation.existing_node)
        tree._nodes.append(mutation.updated_node)
        tree._nodes.remove(mutation.existing_node.left_child)
        tree._nodes.remove(mutation.existing_node.right_child)

    if mutation.kind == "grow":
        tree._nodes.remove(mutation.existing_node)
        tree._nodes.append(mutation.updated_node.left_child)
        tree._nodes.append(mutation.updated_node.right_child)
        tree._nodes.append(mutation.updated_node)

    for node in tree.nodes:
        if node.right_child == mutation.existing_node:
            node._right_child = mutation.updated_node
        if node.left_child == mutation.existing_node:
            node._left_child = mutation.updated_node


if __name__ == "__main__":
    import doctest
    doctest.testmod()