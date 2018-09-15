import numpy as np

from bartpy.model import Model
from bartpy.node import LeafNode


class LeafNodeSampler:

    def __init__(self, model: Model, node: LeafNode):
        self.model = model
        self.node = node

    def step(self):
        self.node.set_value(self.sample())

    def sample(self) -> float:
        prior_var = self.model.sigma_m ** 2
        n = self.node.data.n_obsv
        likihood_var = (self.model.sigma.current_value() ** 2) / n
        likihood_mean = np.mean(self.node.data.y)
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return np.random.normal(posterior_mean, np.power(posterior_variance / self.model.n_trees, 0.5))