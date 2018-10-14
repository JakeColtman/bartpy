import numpy as np

from bartpy.model import Model
from bartpy.node import LeafNode
from bartpy.samplers.sampler import Sampler


class LeafNodeSampler(Sampler):
    """
    Responsible for generating samples of the leaf node predictions
    Essentially just draws from a normal distribution with prior specified by model parameters

    Uses a cache of draws from a normal(0, 1) distribution to improve sampling performance
    """

    def __init__(self):
        self.random_samples = list(np.random.normal(size=50000))

    def step(self, model: Model, node: LeafNode):
        node.set_value(self.sample(model, node))

    def get_next_rand(self):
        if len(self.random_samples) == 0:
            self.random_samples = list(np.random.normal(size=50000))
        return self.random_samples.pop()

    def sample(self, model: Model, node: LeafNode) -> float:
        prior_var = model.sigma_m ** 2
        n = node.data.n_obsv
        likihood_var = (model.sigma.current_value() ** 2) / n
        likihood_mean = np.mean(node.data.y)
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return posterior_mean + (self.get_next_rand() * np.power(posterior_variance / model.n_trees, 0.5))