from typing import Callable, List, Mapping

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from bartpy.node import sample_split_node, LeafNode, DecisionNode
from bartpy.samplers.treemutation.proposer import TreeMutationProposer
from bartpy.tree import Tree


def uniformly_sample_grow_mutation(tree: Tree) -> TreeMutation:
    node = random_splittable_leaf_node(tree)
    updated_node = sample_split_node(node)
    return GrowMutation(node, updated_node)


def uniformly_sample_prune_mutation(tree: Tree) -> TreeMutation:
    node = random_prunable_decision_node(tree)
    updated_node = LeafNode(node.split, depth=node.depth)
    return PruneMutation(node, updated_node)


class UniformMutationProposer(TreeMutationProposer):

    def __init__(self, prob_method: List[float]=None, prob_method_lookup: Mapping[Callable[[Tree], TreeMutation], float]=None):
        if prob_method_lookup is not None:
            self.prob_method_lookup = prob_method_lookup
        else:
            if prob_method is None:
                prob_method = [0.5, 0.5]
            self.prob_method_lookup = {x[0]: x[1] for x in zip([uniformly_sample_grow_mutation, uniformly_sample_prune_mutation], prob_method)}

    def sample_mutation_method(self) -> Callable[[Tree], TreeMutation]:
        return np.random.choice(list(self.prob_method_lookup.keys()), p=list(self.prob_method_lookup.values()))

    def propose(self, tree_structure: Tree) -> TreeMutation:
        method = self.sample_mutation_method()
        try:
            return method(tree_structure)
        except NoSplittableVariableException:
            return self.propose(tree_structure)
        except NoPrunableNodeException:
            return self.propose(tree_structure)


def random_splittable_leaf_node(tree: Tree) -> LeafNode:
    """
    Returns a random leaf node that can be split in a non-degenerate way
    i.e. a random draw from the set of leaf nodes that have at least two distinct values in their covariate matrix
    """
    splittable_nodes = tree.splittable_leaf_nodes
    if len(splittable_nodes) > 0:
        return np.random.choice(splittable_nodes)
    else:
        raise NoSplittableVariableException()


def random_prunable_decision_node(tree: Tree) -> DecisionNode:
    """
    Returns a random decision node that can be pruned
    i.e. a random draw from the set of decision nodes that have two leaf node children
    """
    leaf_parents = tree.prunable_decision_nodes
    if len(leaf_parents) == 0:
        raise NoPrunableNodeException()
    return np.random.choice(leaf_parents)