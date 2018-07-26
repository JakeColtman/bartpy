from abc import abstractclassmethod
from collections import namedtuple
from copy import deepcopy

from bartpy.tree import LeafNode, TreeStructure, split_node


Proposal = namedtuple("Proposal", ["existing_node", "updated_node"])


class Mutation:

    def __init__(self, tree_structure: TreeStructure):
        self.tree_structure = tree_structure

    @abstractclassmethod
    def proposal(self) -> Proposal:
        raise NotImplementedError()

    def apply(self):
        proposal = self.proposal()
        self.tree_structure.update_node(proposal.existing_node, proposal.updated_node)


class GrowMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_node()
        updated_node = split_node(node)
        return Proposal(node, updated_node)


class PruneMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        updated_node.update_left_child(None)
        updated_node.update_right_child(None)
        return Proposal(node, updated_node)


class ChangeMutation(Mutation):

    def __init__(self, tree_structure: TreeStructure):
        super().__init__(tree_structure)

    def proposal(self) -> Proposal:
        node = self.tree_structure.random_leaf_parent()
        updated_node = deepcopy(node)
        leaf_node = LeafNode(updated_node.data)
        updated_split_node = split_node(leaf_node)
        return Proposal(node, updated_split_node)
