from abc import abstractclassmethod
from copy import deepcopy
from typing import Callable

from bartpy.tree import TreeNode, LeafNode, TreeStructure, split_node


class Mutation:

    def __init__(self, tree_structure: TreeStructure):
        self.tree_structure = tree_structure

    @abstractclassmethod
    def apply(self) -> None:
        raise NotImplementedError()


class GrowMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def apply(self):
        node = self.tree_structure.random_leaf_node()
        updated_node = split_node(node)
        self.tree_structure.update_node(node, updated_node)


class PruneMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def apply(self) -> None:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        updated_node.update_left_child(None)
        updated_node.update_right_child(None)
        self.tree_structure.update_node(node, updated_node)


class ChangeMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def apply(self) -> None:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        leaf_node = LeafNode(updated_node.data)
        updated_split_node = split_node(leaf_node)
        self.tree_structure.update_node(node, updated_split_node)