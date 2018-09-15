import numpy as np
import pandas as pd

from bartpy.model import Model
from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.sampler import Sampler
from bartpy.proposer import Proposer


class SklearnModel:

    def __init__(self,
                 n_trees: int=50,
                 sigma: Sigma=Sigma(100., 0.001),
                 n_samples: int=200,
                 n_burn: int=200,
                 p_grow: float=0.5,
                 p_prune: float=0.5,
                 alpha: float=0.95,
                 beta: float=2.):
        self.n_trees = n_trees
        self.sigma = sigma
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.alpha = alpha
        self.beta = beta
        self.data, self.model, self.proposer, self.sampler, self.samples = [None] * 5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SklearnModel':
        self.data = Data(X, y, normalize=True)
        self.model = Model(self.data, self.sigma, n_trees=self.n_trees, alpha=self.alpha, beta=self.beta)
        self.proposer = Proposer(self.p_grow, self.p_prune)
        self.sampler = Sampler(self.model, self.proposer)
        self.samples = self.sampler.samples(self.n_samples, self.n_burn)
        return self

    def predict(self, X: np.ndarray=None):
        if X is not None:
            raise NotImplementedError("Out of sample prediction not supported")
        return self.data.unnormalize_y(self.samples.mean(axis=0))
