from abc import abstractclassmethod
from typing import Callable

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation, PruneMutation, GrowMutation
from bartpy.node import sample_split_node
from bartpy.tree import LeafNode, Tree, random_prunable_decision_node, random_splittable_leaf_node


class TreeMutationProposer:

    def __init__(self, tree_structure: Tree):
        self.tree_structure = tree_structure

    @abstractclassmethod
    def proposal(self) -> TreeMutation:
        raise NotImplementedError()


class GrowTreeMutationProposer(TreeMutationProposer):

    def __init__(self, tree_structure: Tree):
        super().__init__(tree_structure)

    def proposal(self) -> TreeMutation:
        node = random_splittable_leaf_node(self.tree_structure)
        updated_node = sample_split_node(node)
        return GrowMutation(node, updated_node)


class PruneTreeMutationProposer(TreeMutationProposer):

    def __init__(self, tree_structure: Tree):
        super().__init__(tree_structure)

    def proposal(self) -> TreeMutation:
        node = random_prunable_decision_node(self.tree_structure)
        updated_node = LeafNode(node.split, depth=node.depth)
        return PruneMutation(node, updated_node)


class Proposer:

    def __init__(self, p_grow: float, p_prune: float):
        self.p_grow = p_grow
        self.p_prune = p_prune

    def sample_mutation_method(self) -> Callable[[Tree], TreeMutationProposer]:
        method = np.random.choice([GrowTreeMutationProposer, PruneTreeMutationProposer], p=[self.p_grow, self.p_prune])
        return method

    def propose(self, tree_structure: Tree) -> TreeMutation:
        method = self.sample_mutation_method()
        try:
            return method(tree_structure).proposal()
        except NoSplittableVariableException:
            return self.propose(tree_structure)
        except NoPrunableNodeException:
            return self.propose(tree_structure)
