import numpy as np
from scipy.stats import invgamma

from bartpy.model import Model
from bartpy.sigma import Sigma


class SigmaSampler:

    def __init__(self, model: Model, sigma: Sigma):
        self.model = model
        self.sigma = sigma

    def step(self) -> None:
        self.sigma.set_value(self.sample())

    def sample(self) -> float:
        return 0.1
        posterior_alpha = self.sigma.alpha + (self.model.data.n_obsv / 2.)
        posterior_beta = self.sigma.beta + (0.5 * (np.sum(np.power(self.model.residuals(), 2))))
        return np.power(invgamma(posterior_alpha, posterior_beta).rvs(1)[0], 0.5)