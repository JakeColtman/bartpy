from typing import Optional

import numpy as np

from bartpy.model import Model
from bartpy.mutation import TreeMutation
from bartpy.node import LeafNode
from bartpy.samplers.proposer import TreeMutationProposer
from bartpy.sigma import Sigma
from bartpy.tree import Tree, mutate


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


class TreeMutationSampler:

    def __init__(self, model: Model, tree_structure: Tree, proposer: TreeMutationProposer):
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

    def transition_ratio(self, proposal: TreeMutation) -> float:
        return self.proposer.log_transition_ratio(self.tree_structure, proposal)

    def tree_structure_ratio(self, proposal: TreeMutation) -> float:
        return self.proposer.log_tree_structure_ratio(self.model, self.tree_structure, proposal)

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