from bartpy.model import Model
from bartpy.samplers.treemutation.treemutation import TreeMutationSampler
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler


class SampleSchedule:

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: SigmaSampler):
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler

    def steps(self, model: Model):
        for tree in model.refreshed_trees():
            yield lambda: self.tree_sampler.step(model, tree)
            for node in tree.leaf_nodes:
                yield lambda: LeafNodeSampler().step(model, node)
        yield lambda: SigmaSampler().step(model, model.sigma)