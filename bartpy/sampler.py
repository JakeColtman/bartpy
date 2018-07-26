from typing import Optional

import numpy as np

from bartpy.mutation import Proposer, Proposal
from bartpy.tree import TreeStructure


class TreeSampler:

    def __init__(self, tree_structure: TreeStructure, proposer: Proposer):
        self.proposer = proposer
        self.tree_structure = tree_structure

    def sample(self) -> Optional[Proposal]:
        proposal = self.proposer.propose(self.tree_structure)
        ratio = self.proposal_ratio(proposal)
        if np.random.uniform(0, 1) < ratio:
            return proposal
        else:
            return None

    def proposal_ratio(self, proposal: Proposal):
        return self.transition_ratio(proposal) * self.likihood_ratio(proposal) * self.tree_structure_ratio(proposal)

    def transition_ratio(self, proposal: Proposal):
        if proposal.kind == "grow":
            return self.transition_ratio_grow(proposal)
        elif proposal.kind == "prune":
            return self.transition_ratio_prune(proposal)
        elif proposal.kind == "change":
            return self.transition_ratio_change(proposal)
        else:
            raise NotImplementedError("kind {} not supported".format(proposal.kind))

    def transition_ratio_grow(self, proposal: Proposal):
        prob_grow_node_selected = 1.0 / len(self.tree_structure.leaf_nodes())
        prob_attribute_selected = 1.0 / len(proposal.existing_node.data.variables)
        prob_value_selected_within_attribute = 1.0 / len(proposal.existing_node.data.unique_values(proposal.updated_node.split.splitting_variable))

        prob_grow_selected = prob_grow_node_selected * prob_attribute_selected * prob_value_selected_within_attribute
        prob_prune_selected = 1.0 * len(self.tree_structure.leaf_parents()) + 1
        prob_selection_ratio = prob_prune_selected / prob_grow_selected
        prune_grow_ratio = self.proposer.p_prune / self.proposer.p_grow

        return prune_grow_ratio * prob_selection_ratio

    def transition_ratio_prune(self, proposal: Proposal):
        prob_grow_node_selected = 1.0 / (len(self.tree_structure.leaf_nodes()) - 1)
        prob_attribute_selected = 1.0 / len(proposal.updated_node.data.variables)
        prob_value_selected_within_attribute = 1.0 / len(proposal.updated_node.data.unique_values(proposal.existing_node.split.splitting_variable))

        prob_grow_selected = prob_grow_node_selected * prob_attribute_selected * prob_value_selected_within_attribute
        prob_prune_selected = 1.0 * len(self.tree_structure.leaf_parents())
        prob_selection_ratio = prob_grow_selected / prob_prune_selected
        grow_prune_ratio = self.proposer.p_grow / self.proposer.p_prune

        return grow_prune_ratio * prob_selection_ratio

    def transition_ratio_change(self, proposal: Proposal) -> float:
        prob_value_selected_within_attribute_existing = 1.0 / len(proposal.existing_node.data.unique_values(proposal.existing_node.split.splitting_variable))
        prob_value_selected_within_attribute_updated = 1.0 / len(proposal.updated_node.data.unique_values(proposal.updated_node.split.splitting_variable))
        return prob_value_selected_within_attribute_updated / prob_value_selected_within_attribute_existing

    def tree_structure_ratio(self, proposal: Proposal):
        return 1.0

    def likihood_ratio(self, proposal: Proposal):
        return 1.0