import numpy as np

from bartpy.model import Model
from bartpy.sigma import Sigma


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def step(self) -> None:
        self.sigma.set_value(self.sample())

    def sample(self) -> float:
        posterior_alpha = self.sigma.alpha + (self.model.data.n_obsv / 2.)
        posterior_beta = self.sigma.beta + (0.5 * (np.sum(np.square(self.model.residuals()))))
        draw = np.power(np.random.gamma(posterior_alpha, 1./posterior_beta), -0.5)
        return draw
