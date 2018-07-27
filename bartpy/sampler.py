from typing import Optional

import numpy as np

from bartpy.model import Model
from bartpy.mutation import Proposer, TreeMutation
from bartpy.tree import TreeStructure, LeafNode
from bartpy.sigma import Sigma


def likihood_node(node: LeafNode, sigma: Sigma, sigma_mu: float) -> float:
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)

    n = node.data.n_obsv

    first_term = np.power(2 * np.pi * var, - n / 2.)
    second_term = np.power(var / (var + n * var_mu), 0.5)
    third_term = - (1 / (2 * var))
    residuals = node.residuals()
    mean_residual = np.mean(residuals)
    sum_sq_error = np.sum(np.power(residuals - mean_residual, 2))

    fourth_term = (np.power(mean_residual, 2) * np.power(n, 2)) / (n + (var / var_mu))
    fifth_term = n * np.power(mean_residual, 2)

    return first_term * second_term * np.exp(third_term * (sum_sq_error - fourth_term + fifth_term))


class TreeMutationSampler:

    def __init__(self, model: Model, tree_structure: TreeStructure, proposer: Proposer):
        self.proposer = proposer
        self.tree_structure = tree_structure
        self.model = model

    def sample(self) -> Optional[TreeMutation]:
        proposal = self.proposer.propose(self.tree_structure)
        ratio = self.proposal_ratio(proposal)
        if np.random.uniform(0, 1) < ratio:
            return proposal
        else:
            return None

    def proposal_ratio(self, proposal: TreeMutation):
        return self.transition_ratio(proposal) * self.likihood_ratio(proposal) * self.tree_structure_ratio(proposal)

    def transition_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.transition_ratio_grow(proposal)
        elif proposal.kind == "prune":
            return self.transition_ratio_prune(proposal)
        elif proposal.kind == "change":
            return self.transition_ratio_change(proposal)
        else:
            raise NotImplementedError("kind {} not supported".format(proposal.kind))

    def transition_ratio_grow(self, proposal: TreeMutation):
        prob_grow_node_selected = 1.0 / len(self.tree_structure.leaf_nodes())
        prob_attribute_selected = 1.0 / len(proposal.existing_node.data.variables)
        prob_value_selected_within_attribute = 1.0 / len(proposal.existing_node.data.unique_values(proposal.updated_node.split.splitting_variable))

        prob_grow_selected = prob_grow_node_selected * prob_attribute_selected * prob_value_selected_within_attribute
        prob_prune_selected = 1.0 * len(self.tree_structure.leaf_parents()) + 1
        prob_selection_ratio = prob_prune_selected / prob_grow_selected
        prune_grow_ratio = self.proposer.p_prune / self.proposer.p_grow

        return prune_grow_ratio * prob_selection_ratio

    def transition_ratio_prune(self, proposal: TreeMutation):
        prob_grow_node_selected = 1.0 / (len(self.tree_structure.leaf_nodes()) - 1)
        prob_attribute_selected = 1.0 / len(proposal.updated_node.data.variables)
        prob_value_selected_within_attribute = 1.0 / len(proposal.updated_node.data.unique_values(proposal.existing_node.split.splitting_variable))

        prob_grow_selected = prob_grow_node_selected * prob_attribute_selected * prob_value_selected_within_attribute
        prob_prune_selected = 1.0 * len(self.tree_structure.leaf_parents())
        prob_selection_ratio = prob_grow_selected / prob_prune_selected
        grow_prune_ratio = self.proposer.p_grow / self.proposer.p_prune

        return grow_prune_ratio * prob_selection_ratio

    def transition_ratio_change(self, proposal: TreeMutation) -> float:
        return 1.0

    def tree_structure_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.likihood_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.likihood_ratio_prune(proposal)
        if proposal.kind == "change":
            return self.tree_structure_ratio_change(proposal)

    def tree_structure_ratio_change(self, proposal: TreeMutation):
        return 1.0

    def likihood_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.likihood_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.likihood_ratio_prune(proposal)
        if proposal.kind == "change":
            return self.likihood_ratio_change(proposal)

    def likihood_ratio_grow(self, proposal: TreeMutation):
        left_child_likihood = likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        numerator = left_child_likihood * right_child_likihood
        denom = likihood_node(proposal.existing_node, self.model.sigma, self.model.sigma_m)
        return numerator / denom

    def likihood_ratio_prune(self, proposal: TreeMutation):
        numerator = likihood_node(proposal.updated_node, self.model.sigma, self.model.sigma_m)
        left_child_likihood = likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        denom = left_child_likihood * right_child_likihood
        return numerator / denom

    def likihood_ratio_change(self, proposal: TreeMutation):
        left_child_likihood = likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        denom = left_child_likihood * right_child_likihood

        left_child_likihood = likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        numerator = left_child_likihood * right_child_likihood
        return numerator / denom


class Sampler:

    def __init__(self, model: Model, proposer: Proposer):
        self.model = model
        self.proposer = proposer

    def sample(self):
        for tree in self.model.trees:
            tree_sampler = TreeMutationSampler(self.model, tree, self.proposer)
            tree_mutation = tree_sampler.sample()
            tree.update_node(tree_mutation)
