from typing import List

import numpy as np

from bartpy.mutation import TreeMutation
from bartpy.node import TreeNode, LeafNode, DecisionNode


class Tree:
    """
    An encapsulation of the structure of a single decision tree
    Contains no logic, but keeps track of 4 different kinds of nodes within the tree:
      - leaf nodes
      - decision nodes
      - splittable leaf nodes
      - prunable decision nodes
    """

    def __init__(self, nodes: List[TreeNode]):
        self._nodes = nodes
        self.cache_up_to_date = False
        self._prediction = np.zeros_like(self._nodes[0]._split._data._y)

    @property
    def nodes(self) -> List[TreeNode]:
        """
        List of all nodes contained in the tree
        """
        return self._nodes

    @property
    def leaf_nodes(self) -> List[LeafNode]:
        """
        List of all of the leaf nodes in the tree
        """
        return [x for x in self._nodes if x.is_leaf_node()]

    @property
    def splittable_leaf_nodes(self) -> List[LeafNode]:
        """
        List of all leaf nodes in the tree which can be split in a non-degenerate way
        i.e. not all rows of the covariate matrix are duplicates
        """
        return [x for x in self.leaf_nodes if x.is_splittable()]

    @property
    def decision_nodes(self) -> List[DecisionNode]:
        """
        List of decision nodes in the tree.
        Decision nodes are internal split nodes, i.e. not leaf nodes
        """
        return [x for x in self._nodes if x.is_decision_node()]

    @property
    def prunable_decision_nodes(self) -> List[DecisionNode]:
        """
        List of decision nodes in the tree that are suitable for pruning
        In particular, decision nodes that have two leaf node children
        """
        return [x for x in self.decision_nodes if x.is_prunable()]

    def update_y(self, y: np.ndarray) -> None:
        """
        Update the cached value of the target array in all nodes
        Used to pass in the residuals from the sum of all of the other trees
        """
        self.cache_up_to_date = False
        for node in self.nodes:
            node.split.update_y(y)

    def predict(self) -> np.ndarray:
        """
        Generate a set of predictions with the same dimensionality as the target array
        Note that the prediction is from one tree, so represents only (1 / number_of_trees) of the target
        """
        if self.cache_up_to_date:
            return self._prediction
        for leaf in self.leaf_nodes:
            self._prediction[leaf.split.condition()] = leaf.predict()
        self.cache_up_to_date = True
        return self._prediction

    def out_of_sample_predict(self, X) -> np.ndarray:
        prediction = np.array([0.] * len(X))
        for leaf in self.leaf_nodes:
            prediction[leaf.split.out_of_sample_condition(X)] = leaf.predict()
        return prediction

    def remove_node(self, node: TreeNode) -> None:
        """
        Remove a single node from the tree
        Note that this is non-recursive, only drops the node and not any children
        """
        self._nodes.remove(node)

    def add_node(self, node: TreeNode) -> None:
        """
        Add a node to the tree
        Note that this is non-recursive, only adds the node and not any children
        """
        self._nodes.append(node)


def mutate(tree: Tree, mutation: TreeMutation) -> None:
    """
    Apply a change to the structure of the tree
    Modifies not only the tree, but also the links between the TreeNodes
    """
    tree.cache_up_to_date = False

    if mutation.kind == "prune":
        tree.remove_node(mutation.existing_node)
        tree.remove_node(mutation.existing_node.left_child)
        tree.remove_node(mutation.existing_node.right_child)
        tree.add_node(mutation.updated_node)

    if mutation.kind == "grow":
        tree.remove_node(mutation.existing_node)
        tree.add_node(mutation.updated_node.left_child)
        tree.add_node(mutation.updated_node.right_child)
        tree.add_node(mutation.updated_node)

    for node in tree.nodes:
        if node.right_child == mutation.existing_node:
            node._right_child = mutation.updated_node
        if node.left_child == mutation.existing_node:
            node._left_child = mutation.updated_node
