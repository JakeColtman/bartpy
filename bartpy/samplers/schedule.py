from typing import Callable, Generator
from bartpy.model import Model
from bartpy.samplers.treemutation.treemutation import TreeMutationSampler
from bartpy.samplers.leafnode import LeafNodeSampler
from bartpy.samplers.sigma import SigmaSampler
from bartpy.samplers.sampler import Sampler


class SampleSchedule:
    """
    The SampleSchedule class is responsible for handling the ordering of sampling within a Gibbs step
    It is useful to encapsulate this logic if we wish to expand the model

    Parameters
    ----------
    tree_sampler: TreeMutationSampler
        How to sample tree mutation space
    leaf_sampler: LeafNodeSampler
        How to sample leaf node predictions
    sigma_sampler: SigmaSampler
        How to sample sigma values
    """

    def __init__(self,
                 tree_sampler: TreeMutationSampler,
                 leaf_sampler: LeafNodeSampler,
                 sigma_sampler: SigmaSampler):
        self.leaf_sampler = leaf_sampler
        self.sigma_sampler = sigma_sampler
        self.tree_sampler = tree_sampler

    def steps(self, model: Model) -> Generator[Callable[[Model], Sampler], None, None]:
        """
        Create a generator of the steps that need to be called to complete a full Gibbs sample

        Parameters
        ----------
        model: Model
            The model being sampled

        Returns
        -------
        Generator[Callable[[Model], Sampler], None, None]
            A generator a function to be called
        """
        for tree in model.refreshed_trees():
            yield lambda: self.tree_sampler.step(model, tree)
            for node in tree.leaf_nodes:
                yield lambda: self.leaf_sampler.step(model, node)
        yield lambda: self.sigma_sampler.step(model, model.sigma)
