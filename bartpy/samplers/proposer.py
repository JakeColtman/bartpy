from abc import abstractmethod
from typing import Callable, Mapping, Any

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.model import Model
from bartpy.mutation import TreeMutation, PruneMutation, GrowMutation
from bartpy.node import sample_split_node, TreeNode, LeafNode, DecisionNode
from bartpy.tree import LeafNode, Tree


def log_probability_node_split(model: Model, node: TreeNode):
    return np.log(model.alpha * np.power(1 + node.depth, -model.beta))


def log_probability_node_not_split(model: Model, node: TreeNode):
    return np.log(1. - model.alpha * np.power(1 + node.depth, -model.beta))


def log_probability_split_within_node(mutation: GrowMutation) -> float:
    """
    The probability of a node being split in the given way
    """

    prob_splitting_variable_selected = - np.log(mutation.existing_node.data.n_splittable_variables)
    splitting_variable = mutation.updated_node.variable_split_on().splitting_variable
    prob_value_selected_within_variable = - np.log(mutation.existing_node.data.n_unique_values(splitting_variable))
    return prob_splitting_variable_selected + prob_value_selected_within_variable


def log_probability_split_within_tree(tree_structure: Tree, mutation: GrowMutation) -> float:
    prob_node_chosen_to_split_on = - np.log(n_splittable_leaf_nodes(tree_structure))
    prob_split_chosen = log_probability_split_within_node(mutation)
    return prob_node_chosen_to_split_on + prob_split_chosen


class TreeMutationProposer:

    @abstractmethod
    def propose(self, tree_structure: Tree) -> TreeMutation:
        raise NotImplementedError()

    def log_probability(self, mutation: TreeMutation) -> float:
        raise NotImplementedError()

    def log_transition_ratio(self, mutation: TreeMutation) -> float:
        return self.log_probability(mutation) - self.log_probability(mutation.reverse())

    def log_tree_structure_ratio(self, mutation: TreeMutation) -> float:
        raise NotImplementedError()


class UniformGrowTreeMutationProposer():

    def propose(self, tree_structure: Tree) -> TreeMutation:
        node = random_splittable_leaf_node(tree_structure)
        updated_node = sample_split_node(node)
        return GrowMutation(node, updated_node)


class UniformPruneTreeMutationProposer():

    def propose(self, tree_structure: Tree) -> TreeMutation:
        node = random_prunable_decision_node(tree_structure)
        updated_node = LeafNode(node.split, depth=node.depth)
        return PruneMutation(node, updated_node)


class UniformMutationProposer(TreeMutationProposer):

    def __init__(self, prob_method_lookup: Mapping[Any, float]):
        self.prob_method_lookup = prob_method_lookup

    def sample_mutation_method(self) -> Callable[[Tree], TreeMutationProposer]:
        method = np.random.choice(list(self.prob_method_lookup.keys()), p=list(self.prob_method_lookup.values()))
        return method

    def propose(self, tree_structure: Tree) -> TreeMutation:
        method = self.sample_mutation_method()
        try:
            return method().propose(tree_structure)
        except NoSplittableVariableException:
            return self.propose(tree_structure)
        except NoPrunableNodeException:
            return self.propose(tree_structure)

    def log_transition_ratio(self, tree_structure: Tree, mutation: TreeMutation):
        if mutation.kind == "prune":
            return self.log_prune_transition_ratio(tree_structure, mutation)
        if mutation.kind == "grow":
            return self.log_grow_transition_ratio(tree_structure, mutation)
        else:
            raise NotImplementedError("kind {} not supported".format(mutation.kind))

    def log_grow_transition_ratio(self, tree_structure: Tree, mutation: GrowMutation):
        prob_prune_selected = - np.log(n_prunable_decision_nodes(tree_structure) + 1)
        prob_grow_selected = log_probability_split_within_tree(tree_structure, mutation)

        prob_selection_ratio = prob_prune_selected - prob_grow_selected
        prune_grow_ratio = np.log(self.prob_method_lookup[UniformPruneTreeMutationProposer] / self.prob_method_lookup[UniformGrowTreeMutationProposer])

        return prune_grow_ratio + prob_selection_ratio

    def log_prune_transition_ratio(self, tree_structure: Tree, mutation: PruneMutation):
        prob_grow_node_selected = - np.log(n_splittable_leaf_nodes(tree_structure) - 1)
        prob_split = log_probability_split_within_node(GrowMutation(mutation.updated_node, mutation.existing_node))
        prob_grow_selected = prob_grow_node_selected + prob_split

        prob_prune_selected = - np.log(n_prunable_decision_nodes(tree_structure))

        prob_selection_ratio = prob_grow_selected - prob_prune_selected
        grow_prune_ratio = np.log(self.prob_method_lookup[UniformGrowTreeMutationProposer] / self.prob_method_lookup[UniformPruneTreeMutationProposer])

        return grow_prune_ratio + prob_selection_ratio

    def log_tree_structure_ratio(self, model: Model, tree_structure: Tree, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.log_tree_structure_ratio_grow(model, tree_structure, proposal)
        if proposal.kind == "prune":
            return self.log_tree_structure_ratio_prune(model, proposal)

    def log_tree_structure_ratio_grow(self, model: Model, tree_structure: Tree, proposal: GrowMutation):
        denominator = log_probability_node_not_split(model, proposal.existing_node)

        prob_left_not_split = log_probability_node_not_split(model, proposal.updated_node.left_child)
        prob_right_not_split = log_probability_node_not_split(model, proposal.updated_node.right_child)
        prob_updated_node_split = log_probability_node_split(model, proposal.updated_node)
        prob_chosen_split = log_probability_split_within_tree(tree_structure, proposal)
        numerator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split

        return numerator - denominator

    def log_tree_structure_ratio_prune(self, model: Model, proposal: PruneMutation):
        numerator = log_probability_node_not_split(model, proposal.updated_node)

        prob_left_not_split = log_probability_node_not_split(model, proposal.existing_node.left_child)
        prob_right_not_split = log_probability_node_not_split(model, proposal.existing_node.left_child)
        prob_updated_node_split = log_probability_node_split(model, proposal.existing_node)
        prob_chosen_split = log_probability_split_within_node(GrowMutation(proposal.updated_node, proposal.existing_node))
        denominator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split

        return numerator - denominator


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
        raise NoPrunableNodeException
    return np.random.choice(leaf_parents)


def n_prunable_decision_nodes(tree: Tree) -> int:
    """
    The number of prunable decision nodes
    i.e. how many decision nodes have two leaf children
    """
    return len(tree.prunable_decision_nodes)


def n_splittable_leaf_nodes(tree: Tree) -> int:
    """
    The number of splittable leaf nodes
    i.e. how many leaf nodes have more than one distinct values in their covariate matrix
    """
    return len(tree.splittable_leaf_nodes)