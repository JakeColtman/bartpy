from bartpy.samplers.treemutation import TreeMutationSampler
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler


class SampleSchedule:

    def __init__(self, model, proposer):
        self.model = model
        self.proposer = proposer

    def steps(self):
        for tree in self.model.refreshed_trees():
            yield TreeMutationSampler(self.model, tree, self.proposer)
            for node in tree.leaf_nodes:
                yield LeafNodeSampler(self.model, node)
        yield SigmaSampler(self.model, self.model.sigma)