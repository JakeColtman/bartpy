from typing import Optional

import numpy as np
from scipy.stats import invgamma

from bartpy.model import Model
from bartpy.mutation import TreeMutation, GrowMutation, PruneMutation
from bartpy.node import LeafNode, TreeNode
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


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def step(self) -> None:
        self.sigma.set_value(self.sample())

    def sample(self) -> float:
        return 0.1
        posterior_alpha = self.sigma.alpha + (self.model.data.n_obsv / 2.)
        posterior_beta = self.sigma.beta + (0.5 * (np.sum(np.power(self.model.residuals(), 2))))
        return np.power(invgamma(posterior_alpha, posterior_beta).rvs(1)[0], 0.5)


class LeafNodeSampler:

    def __init__(self, model: Model, node: LeafNode):
        self.model = model
        self.node = node

    def step(self):
        print(self.node.current_value)
        self.node.set_value(self.sample())
        print(self.node.current_value)
        print("----")

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

    def step(self) -> None:
        mutation = self.sample()
        if mutation is not None:
            mutate(self.tree_structure, mutation)

    def proposal_ratio(self, proposal: TreeMutation):
        return self.transition_ratio(proposal) + self.likihood_ratio(proposal) + self.tree_structure_ratio(proposal)

    def transition_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.transition_ratio_grow(proposal)
        elif proposal.kind == "prune":
            return self.transition_ratio_prune(proposal)
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

    def tree_structure_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.tree_structure_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.tree_structure_ratio_prune(proposal)

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

    def likihood_ratio(self, proposal: TreeMutation):
        if proposal.kind == "grow":
            return self.likihood_ratio_grow(proposal)
        if proposal.kind == "prune":
            return self.likihood_ratio_prune(proposal)
        else:
            raise NotImplementedError("Only prune and grow mutations supported")

    def likihood_ratio_grow(self, proposal: TreeMutation):
        return log_grow_ratio(proposal.existing_node, proposal.updated_node.left_child, proposal.updated_node.right_child, self.model.sigma, self.model.sigma_m)

    def likihood_ratio_prune(self, proposal: TreeMutation):
        return 1.0 / log_grow_ratio(proposal.updated_node, proposal.existing_node.left_child, proposal.existing_node.right_child, self.model.sigma, self.model.sigma_m)


class SampleSchedule:

    def __init__(self, model, proposer):
        self.model = model
        self.proposer = proposer

    def steps(self):
        for tree in self.model.trees:
            yield TreeMutationSampler(self.model, tree, self.proposer)
            for node in tree.leaf_nodes:
                yield LeafNodeSampler(self.model, node)
        yield SigmaSampler(self.model, self.model.sigma)


class Sampler:

    def __init__(self, model: Model, schedule: SampleSchedule):
        self.schedule = schedule
        self.model = model

    def step(self):
        for ss in self.schedule.steps():
            print(ss)
            ss.step()

    def samples(self, n_samples: int, n_burn: int) -> np.ndarray:
        for bb in range(n_burn):
            print("Burn - ", bb)
            self.step()
            print([len(x.nodes) for x in self.model.trees])
        trace = []
        for ss in range(n_samples):
            print("Sample - ", ss)
            self.step()
            trace.append(self.model.predict())
        return np.array(trace)
