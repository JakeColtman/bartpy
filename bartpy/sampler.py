from typing import Optional

import numpy as np
from scipy.stats import invgamma

from bartpy.model import Model
from bartpy.proposer import Proposer
from bartpy.tree import TreeStructure, LeafNode, TreeMutation
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


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def sample(self) -> float:
        posterior_alpha = self.sigma.alpha + self.model.data.n_obsv
        posterior_beta = self.sigma.beta + (0.5 * (np.sum(self.model.residuals())))
        return invgamma(posterior_alpha, posterior_beta).rvs(1)


class LeafNodeSampler:

    def __init__(self, model: Model, node: LeafNode):
        self.model = model
        self.node = node

    def sample(self) -> float:
        prior_var = self.model.sigma_m ** 2
        n = self.node.data.n_obsv
        likihood_var = (self.model.sigma.current_value() ** 2) / n
        likihood_mean = np.mean(self.node.residuals())

        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return np.random.normal(posterior_mean, posterior_variance)


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
        prob_value_selected_within_attribute = 1.0 / len(proposal.existing_node.data.unique_values(proposal.existing_node.split.splitting_variable))

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

    def step_leaf(self, node: LeafNode) -> None:
        leaf_sampler = LeafNodeSampler(self.model, node)
        node.set_value(leaf_sampler.sample())

    def step_tree(self, tree: TreeStructure) -> None:
        tree_sampler = TreeMutationSampler(self.model, tree, self.proposer)
        tree_mutation = tree_sampler.sample()
        tree.update_node(tree_mutation)

        for node in tree.leaf_nodes():
            self.step_leaf(node)

    def step_sigma(self, sigma: Sigma) -> None:
        sampler = SigmaSampler(self.model, sigma)
        sigma.set_value(sampler.sample())

    def step(self):
        for tree in self.model.trees:
            self.step_tree(tree)
        self.step_sigma(self.model.sigma)

    def samples(self, n_samples: int, n_burn: int) -> np.ndarray:
        for _ in range(n_burn):
            self.step()
        trace = []
        for _ in range(n_samples):
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
    tree.update_node(sample)
    print(tree.leaf_nodes())
    # proposal = prune_proposer.propose(model.trees[0])
    # print(proposal)
    # ratio = sampler.proposal_ratio(proposal)
    # print(proposal)
    # print(ratio)