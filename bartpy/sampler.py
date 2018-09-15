from typing import Optional

import numpy as np
from scipy.stats import invgamma

from bartpy.model import Model
from bartpy.mutation import TreeMutation, GrowMutation, ChangeMutation, PruneMutation
from bartpy.node import DecisionNode, LeafNode, TreeNode
from bartpy.proposer import Proposer
from bartpy.sigma import Sigma
from bartpy.tree import Tree, n_splittable_leaf_nodes, n_prunable_decision_nodes, mutate


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


def log_grow_ratio(combined_node: LeafNode, left_node: LeafNode, right_node: LeafNode, sigma: Sigma, sigma_mu: float):
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)
    n = combined_node.data.n_obsv
    n_l = left_node.data.n_obsv
    n_r = right_node.data.n_obsv

    first_term = (var * (var + n * sigma_mu)) / ((var + n_l * var_mu) * (var + n_r * var_mu))
    first_term = np.log(np.sqrt(first_term))

    left_resp_contribution = np.square(np.sum(left_node.data.y)) / (var + n_l * sigma_mu)
    right_resp_contribution = np.square(np.sum(right_node.data.y)) / (var + n_r * sigma_mu)
    combined_resp_contribution = np.square(np.sum(combined_node.data.y)) / (var + n * sigma_mu)

    resp_contribution = left_resp_contribution + right_resp_contribution - combined_resp_contribution

    return first_term + ((var_mu / (2 * var)) * resp_contribution)


def log_change_ratio(original_left_node: LeafNode, original_right_node: LeafNode, new_left_node: LeafNode, new_right_node: LeafNode, sigma: Sigma, sigma_mu: float):
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)
    vr = var / var_mu

    n_l_o = original_left_node.data.n_obsv
    n_r_o = original_right_node.data.n_obsv
    n_l_n = new_left_node.data.n_obsv
    n_r_n = new_right_node.data.n_obsv

    first_term = (((vr + n_l_o) * (vr + n_r_o)) / ((vr + n_l_n) * (vr + n_r_n)))
    first_term = np.log(np.sqrt(first_term))

    original_left_contribution = np.square(np.sum(original_left_node.data.y)) / (vr * n_l_o)
    original_right_contribution = np.square(np.sum(original_right_node.data.y)) / (vr * n_r_o)
    new_left_contribution = np.square(np.sum(new_left_node.data.y)) / (vr * n_r_n)
    new_right_contribution = np.square(np.sum(new_right_node.data.y)) / (vr * n_r_n)

    return first_term + ((1 / (2 * var)) * (new_left_contribution + new_right_contribution - original_left_contribution - original_right_contribution))


def log_likihood_node(node: LeafNode, sigma: Sigma, sigma_mu: float) -> float:
    var = np.power(sigma.current_value(), 2)
    var_mu = np.power(sigma_mu, 2)

    n = node.data.n_obsv
    first_term = (- n / 2.) * np.log(2 * np.pi * var)
    second_term = 0.5 * np.log(var / (var + n * var_mu))

    third_term = -0.5 / var

    mean_residual = np.mean(node.data.y)
    mean_residual_squared = np.power(mean_residual, 2)
    sum_sq_error = np.sum(np.power(node.data.y - mean_residual, 2))

    fourth_term = (mean_residual_squared * np.power(n, 2)) / (n + (var / var_mu))
    fifth_term = n * mean_residual_squared

    return first_term + second_term + (third_term * (sum_sq_error - fourth_term + fifth_term))


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def sample(self) -> float:
        return 0.1
        posterior_alpha = self.sigma.alpha + (self.model.data.n_obsv / 2.)
        posterior_beta = self.sigma.beta + (0.5 * (np.sum(np.power(self.model.residuals(), 2))))
        return np.power(invgamma(posterior_alpha, posterior_beta).rvs(1)[0], 0.5)


class LeafNodeSampler:

    def __init__(self, model: Model, node: LeafNode):
        self.model = model
        self.node = node

    def sample(self) -> float:
        prior_var = self.model.sigma_m ** 2
        n = self.node.data.n_obsv
        likihood_var = (self.model.sigma.current_value() ** 2) / n
        likihood_mean = np.mean(self.node.data.y)
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return np.random.normal(posterior_mean, np.power(posterior_variance / self.model.n_trees, 0.5))


class TreeMutationSampler:

    def __init__(self, model: Model, tree_structure: Tree, proposer: Proposer):
        self.proposer = proposer
        self.tree_structure = tree_structure
        self.model = model

    def sample(self) -> Optional[TreeMutation]:
        proposal = self.proposer.propose(self.tree_structure)
        ratio = self.proposal_ratio(proposal)
        if np.log(np.random.uniform(0, 1)) < ratio:
            return proposal
        else:
            return None

    def proposal_ratio(self, proposal: TreeMutation):
        return self.transition_ratio(proposal) + self.likihood_ratio(proposal) + self.tree_structure_ratio(proposal)

    def transition_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.transition_ratio_grow(proposal)
        elif proposal.kind == "prune":
            return self.transition_ratio_prune(proposal)
        elif proposal.kind == "change":
            return self.transition_ratio_change(proposal)
        else:
            raise NotImplementedError("kind {} not supported".format(proposal.kind))

    def transition_ratio_grow(self, proposal: GrowMutation):
        prob_prune_selected = - np.log(n_prunable_decision_nodes(self.tree_structure) + 1)
        prob_grow_selected = log_probability_split_within_tree(self.tree_structure, proposal)

        prob_selection_ratio = prob_prune_selected - prob_grow_selected
        prune_grow_ratio = np.log(self.proposer.p_prune / self.proposer.p_grow)

        return prune_grow_ratio + prob_selection_ratio

    def transition_ratio_prune(self, proposal: PruneMutation):
        prob_grow_node_selected = - np.log(n_splittable_leaf_nodes(self.tree_structure) - 1)
        prob_split = log_probability_split_within_node(GrowMutation(proposal.updated_node, proposal.existing_node))
        prob_grow_selected = prob_grow_node_selected + prob_split

        prob_prune_selected = - np.log(n_prunable_decision_nodes(self.tree_structure))

        prob_selection_ratio = prob_grow_selected - prob_prune_selected
        grow_prune_ratio = np.log(self.proposer.p_grow / self.proposer.p_prune)

        return grow_prune_ratio + prob_selection_ratio

    def transition_ratio_change(self, proposal: ChangeMutation) -> float:
        return 0.0

    def tree_structure_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.tree_structure_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.tree_structure_ratio_prune(proposal)
        if proposal.kind == "change":
            return self.tree_structure_ratio_change(proposal)

    def tree_structure_ratio_grow(self, proposal: GrowMutation):
        denominator = log_probability_node_not_split(self.model, proposal.existing_node)

        prob_left_not_split = log_probability_node_not_split(self.model, proposal.updated_node.left_child)
        prob_right_not_split = log_probability_node_not_split(self.model, proposal.updated_node.right_child)
        prob_updated_node_split = log_probability_node_split(self.model, proposal.updated_node)
        prob_chosen_split = log_probability_split_within_tree(self.tree_structure, proposal)
        numerator = prob_left_not_split + prob_right_not_split + + prob_updated_node_split + prob_chosen_split

        return numerator - denominator

    def tree_structure_ratio_prune(self, proposal: PruneMutation):
        numerator = log_probability_node_not_split(self.model, proposal.updated_node)

        prob_left_not_split = log_probability_node_not_split(self.model, proposal.existing_node.left_child)
        prob_right_not_split = log_probability_node_not_split(self.model, proposal.existing_node.left_child)
        prob_updated_node_split = log_probability_node_split(self.model, proposal.existing_node)
        prob_chosen_split = log_probability_split_within_node(GrowMutation(proposal.updated_node, proposal.existing_node))
        denominator = prob_left_not_split + prob_right_not_split + prob_updated_node_split + prob_chosen_split

        return numerator - denominator

    def tree_structure_ratio_change(self, proposal: ChangeMutation):
        return 0.0

    def likihood_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.likihood_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.likihood_ratio_prune(proposal)
        if proposal.kind == "change":
            return self.likihood_ratio_change(proposal)

    def likihood_ratio_grow(self, proposal: TreeMutation):
        return log_grow_ratio(proposal.existing_node, proposal.updated_node.left_child, proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        # left_child_likihood = log_likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        # right_child_likihood = log_likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        # numerator = left_child_likihood + right_child_likihood
        # denom = log_likihood_node(proposal.existing_node, self.model.sigma, self.model.sigma_m)
        # return numerator - denom

    def likihood_ratio_prune(self, proposal: TreeMutation):
        return 1.0 / log_grow_ratio(proposal.updated_node, proposal.existing_node.left_child, proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        # numerator = log_likihood_node(proposal.updated_node, self.model.sigma, self.model.sigma_m)
        # left_child_likihood = log_likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        # right_child_likihood = log_likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        # denom = left_child_likihood + right_child_likihood
        # return numerator - denom

    def likihood_ratio_change(self, proposal: TreeMutation):
        return log_change_ratio(proposal.existing_node.left_child,
                                proposal.existing_node.right_child,
                                proposal.updated_node.left_child,
                                proposal.updated_node.right_child,
                                self.model.sigma,
                                self.model.sigma_m)
        # left_child_likihood = log_likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        # right_child_likihood = log_likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        # denom = left_child_likihood + right_child_likihood
        #
        # left_child_likihood = log_likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        # right_child_likihood = log_likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        # numerator = left_child_likihood + right_child_likihood
        # return numerator - denom


class Sampler:

    def __init__(self, model: Model, proposer: Proposer):
        self.model = model
        self.proposer = proposer

    def step_leaf(self, node: LeafNode) -> None:
        leaf_sampler = LeafNodeSampler(self.model, node)
        node.set_value(leaf_sampler.sample())

    def step_tree(self, tree: Tree) -> None:
        tree_sampler = TreeMutationSampler(self.model, tree, self.proposer)
        tree_mutation = tree_sampler.sample()
        if tree_mutation is not None:
            mutate(tree, tree_mutation)

    def step_sigma(self, sigma: Sigma) -> None:
        sampler = SigmaSampler(self.model, sigma)
        sigma.set_value(sampler.sample())

    def step(self):
        for tree in self.model.refreshed_trees():
            self.step_tree(tree)
            for node in tree.leaf_nodes:
                self.step_leaf(node)
        self.step_sigma(self.model.sigma)

    def samples(self, n_samples: int, n_burn: int) -> np.ndarray:
        for bb in range(n_burn):
            self.step()
        trace = []
        for ss in range(n_samples):
            self.step()
            trace.append(self.model.predict())
        return np.array(trace)


if __name__ == "__main__":
    from bartpy.data import Data
    import pandas as pd
    from bartpy.sigma import Sigma
    from bartpy.model import Model

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma)

    prune_proposer = Proposer(0.5, 0.5, 0)

    sampler = TreeMutationSampler(model, model.trees[0], prune_proposer)
    sample = None
    while sample is None:
        sample = sampler.sample()

    print(sample)
    tree = model.trees[0]
    print(tree.leaf_nodes())
    tree.mutate(sample)
    print(tree.leaf_nodes())
    # proposal = prune_proposer.propose(model.trees[0])
    # print(proposal)
    # ratio = sampler.proposal_ratio(proposal)
    # print(proposal)
    # print(ratio)