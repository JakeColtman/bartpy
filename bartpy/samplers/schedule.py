from bartpy.samplers.treemutation.treemutation import TreeMutationSampler
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler


class SampleSchedule:

    def __init__(self, model, tree_sampler: TreeMutationSampler):
        self.model = model
        self.tree_sampler = tree_sampler

    def steps(self):
        for tree in self.model.refreshed_trees():
            yield lambda: self.tree_sampler.step(self.model, tree)
            for node in tree.leaf_nodes:
                yield lambda: LeafNodeSampler().step(self.model, node)
        yield lambda: SigmaSampler().step(self.model, self.model.sigma)