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
                 p_grow: float=0.2,
                 p_prune: float=0.2,
                 p_change: float=0.6):
        self.n_trees = n_trees
        self.sigma = sigma
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = p_grow
        self.p_change = p_change
        self.p_prune = p_prune
        self.data, self.model, self.proposer, self.sampler, self.samples = [None] * 5

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SklearnModel':
        self.data = Data(X, y, normalize=True)
        self.model = Model(self.data, self.sigma, n_trees=self.n_trees)
        self.proposer = Proposer(self.p_grow, self.p_prune, self.p_change)
        self.sampler = Sampler(self.model, self.proposer)
        self.samples = self.sampler.samples(self.n_samples, self.n_burn)
        return self

    def predict(self, X: np.ndarray=None):
        return self.data.unnormalize_y(self.samples.mean(axis=0))
