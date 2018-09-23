import numpy as np

from bartpy.model import Model
from bartpy.sigma import Sigma
from bartpy.samplers.sampler import Sampler


class SigmaSampler(Sampler):

    def step(self, model: Model, sigma: Sigma) -> None:
        sigma.set_value(self.sample(model, sigma))

    def sample(self, model: Model, sigma: Sigma) -> float:
        posterior_alpha = sigma.alpha + (model.data.n_obsv / 2.)
        posterior_beta = sigma.beta + (0.5 * (np.sum(np.square(model.residuals()))))
        draw = np.power(np.random.gamma(posterior_alpha, 1./posterior_beta), -0.5)
        return draw
