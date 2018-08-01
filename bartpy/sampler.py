from typing import Optional

import numpy as np
from scipy.stats import invgamma

from bartpy.model import Model
from bartpy.proposer import Proposer
from bartpy.tree import SplitNode, TreeStructure, LeafNode, TreeMutation, GrowMutation, ChangeMutation, PruneMutation
from bartpy.sigma import Sigma


def log_probability_node_split(model: Model, node: SplitNode):
    return np.log(model.alpha * np.power(1 + node.depth, -model.beta))


def log_probability_node_not_split(model: Model, node: SplitNode):
    return np.log(1. - model.alpha * np.power(1 + node.depth, -model.beta))


def log_probability_split_within_node(mutation: GrowMutation) -> float:
    """
    The probability of a node being split in the given way
    """

    prob_splitting_variable_selected = - np.log(mutation.existing_node.data.n_splittable_variables)
    splitting_variable = mutation.updated_node.children_split().splitting_variable
    prob_value_selected_within_variable = - np.log(mutation.existing_node.data.n_unique_values(splitting_variable))
    return prob_splitting_variable_selected + prob_value_selected_within_variable


def log_probability_split_within_tree(tree_structure: TreeStructure, mutation: GrowMutation) -> float:
    prob_node_chosen_to_split_on = - np.log(tree_structure.n_leaf_nodes())
    prob_split_chosen = log_probability_split_within_node(mutation)
    return prob_node_chosen_to_split_on + prob_split_chosen


def log_likihood_node(node: LeafNode, sigma: Sigma, sigma_mu: float) -> float:
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

    return np.log(first_term * second_term * np.exp(third_term * (sum_sq_error - fourth_term + fifth_term)))


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def sample(self) -> float:
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
        return np.exp(self.transition_ratio(proposal) + self.likihood_ratio(proposal) + self.tree_structure_ratio(proposal))

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
        prob_grow_selected = log_probability_split_within_tree(self.tree_structure, proposal)
        prob_prune_selected = - np.log(self.tree_structure.n_leaf_parents() + 1)

        prob_selection_ratio = prob_prune_selected - prob_grow_selected
        prune_grow_ratio = np.log(self.proposer.p_prune / self.proposer.p_grow)

        return prune_grow_ratio + prob_selection_ratio

    def transition_ratio_prune(self, proposal: PruneMutation):
        prob_grow_node_selected = - np.log(self.tree_structure.n_leaf_nodes() - 1)
        prob_split = log_probability_split_within_node(GrowMutation(proposal.updated_node, proposal.existing_node))
        prob_grow_selected = prob_grow_node_selected + prob_split

        prob_prune_selected = - np.log(self.tree_structure.n_leaf_parents())

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
        prob_right_not_split = log_probability_node_not_split(self.model, proposal.updated_node.left_child)
        prob_chosen_split = log_probability_split_within_tree(self.tree_structure, proposal)
        numerator = prob_left_not_split + prob_right_not_split + prob_chosen_split

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
        left_child_likihood = log_likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = log_likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        numerator = left_child_likihood + right_child_likihood
        denom = log_likihood_node(proposal.existing_node, self.model.sigma, self.model.sigma_m)
        return numerator - denom

    def likihood_ratio_prune(self, proposal: TreeMutation):
        numerator = log_likihood_node(proposal.updated_node, self.model.sigma, self.model.sigma_m)
        left_child_likihood = log_likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = log_likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        denom = left_child_likihood + right_child_likihood
        return numerator - denom

    def likihood_ratio_change(self, proposal: TreeMutation):
        left_child_likihood = log_likihood_node(proposal.existing_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = log_likihood_node(proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)
        denom = left_child_likihood + right_child_likihood

        left_child_likihood = log_likihood_node(proposal.updated_node.left_child, self.model.sigma, self.model.sigma_m)
        right_child_likihood = log_likihood_node(proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)
        numerator = left_child_likihood + right_child_likihood
        return numerator - denom


class Sampler:

    def __init__(self, model: Model, proposer: Proposer):
        self.model = model
        self.proposer = proposer

    def step_leaf(self, node: LeafNode) -> None:
        leaf_sampler = LeafNodeSampler(self.model, node)
        node.set_value(leaf_sampler.sample())

    def step_tree(self, tree: TreeStructure) -> None:
        tree_sampler = TreeMutationSampler(self.model, tree, self.proposer)
        tree_mutation = tree_sampler.sample()
        if tree_mutation is not None:
            tree.update_node(tree_mutation)

        for node in tree.leaf_nodes():
            self.step_leaf(node)

    def step_sigma(self, sigma: Sigma) -> None:
        sampler = SigmaSampler(self.model, sigma)
        sigma.set_value(sampler.sample())

    def step(self):
        for tree in self.model.refreshed_trees():
            self.step_tree(tree)
        self.step_sigma(self.model.sigma)

    def samples(self, n_samples: int, n_burn: int) -> np.ndarray:
        for bb in range(n_burn):
            print(bb)

            self.step()
            print(self.model.sigma.current_value())

            print([len(x.nodes()) for x in self.model.trees])
            print([len(set(x.nodes())) for x in self.model.trees])
            print(self.model.residuals())
        trace = []
        for ss in range(n_samples):
            print(ss)
            self.step()
            print([len(x.nodes()) for x in self.model.trees])
            print(self.model.sigma.current_value())
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
    tree.update_node(sample)
    print(tree.leaf_nodes())
    # proposal = prune_proposer.propose(model.trees[0])
    # print(proposal)
    # ratio = sampler.proposal_ratio(proposal)
    # print(proposal)
    # print(ratio)