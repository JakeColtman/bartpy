from abc import abstractclassmethod
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from bartpy.tree import LeafNode, TreeStructure, split_node, TreeNode


class Proposal:

    def __init__(self, kind: str, existing_node: TreeNode, updated_node: Optional[TreeNode]):
        if kind not in ["grow", "change", "prune"]:
            raise NotImplementedError("{} is not a supported proposal".format(kind))
        self.kind = kind
        self.existing_node = existing_node
        self.updated_node = updated_node


class Mutation:

    def __init__(self, tree_structure: TreeStructure):
        self.tree_structure = tree_structure

    @abstractclassmethod
    def proposal(self) -> Proposal:
        raise NotImplementedError()


class GrowMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_node()
        updated_node = split_node(node)
        return Proposal("grow", node, updated_node)


class PruneMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        updated_node.update_left_child(None)
        updated_node.update_right_child(None)
        return Proposal("prune", node, updated_node)


class ChangeMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        leaf_node = LeafNode(updated_node.data)
        updated_split_node = split_node(leaf_node)
        return Proposal("change", node, updated_split_node)


class Proposer:

    def __init__(self, p_grow: float, p_prune: float, p_change: float):
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.p_change = p_change

    def sample_mutation_method(self) -> Callable[[TreeStructure], Mutation]:
        method = np.random.choice([ChangeMutation, GrowMutation, PruneMutation], [self.p_change, self.p_grow, self.p_prune])
        return method

    def propose(self, tree_structure: TreeStructure) -> Proposal:
        method = self.sample_mutation_method()
        return method(tree_structure).proposal()
