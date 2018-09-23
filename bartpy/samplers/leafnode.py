import numpy as np

from bartpy.model import Model
from bartpy.node import LeafNode
from bartpy.samplers.sampler import Sampler


class LeafNodeSampler(Sampler):

    def step(self, model: Model, node: LeafNode):
        node.set_value(self.sample(model, node))

    def sample(self, model: Model, node: LeafNode) -> float:
        prior_var = model.sigma_m ** 2
        n = node.data.n_obsv
        likihood_var = (model.sigma.current_value() ** 2) / n
        likihood_mean = np.mean(node.data.y)
        posterior_variance = 1. / (1. / prior_var + 1. / likihood_var)
        posterior_mean = likihood_mean * (prior_var / (likihood_var + prior_var))
        return np.random.normal(posterior_mean, np.power(posterior_variance / model.n_trees, 0.5))