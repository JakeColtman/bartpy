from typing import Optional

import numpy as np

from bartpy.model import Model
from bartpy.mutation import TreeMutation
from bartpy.samplers.treemutation.proposer import TreeMutationProposer
from bartpy.samplers.treemutation.likihoodratio import TreeMutationLikihoodRatio
from bartpy.tree import Tree, mutate


class TreeMutationSampler:

    def __init__(self, proposer: TreeMutationProposer, likihood_ratio: TreeMutationLikihoodRatio):
        self.proposer = proposer
        self.likihood_ratio = likihood_ratio

    def sample(self, model: Model, tree: Tree) -> Optional[TreeMutation]:
        proposal = self.proposer.propose(tree)
        ratio = self.likihood_ratio.log_probability_ratio(model, tree, proposal)
        if np.log(np.random.uniform(0, 1)) < ratio:
            return proposal
        else:
            return None

    def step(self, model: Model, tree: Tree) -> None:
        mutation = self.sample(model, tree)
        if mutation is not None:
            mutate(tree, mutation)
