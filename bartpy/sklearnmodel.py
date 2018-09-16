import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator

from bartpy.model import Model
from bartpy.data import Data
from bartpy.samplers.schedule import SampleSchedule
from bartpy.samplers.sampler import Sampler
from bartpy.sigma import Sigma
from bartpy.samplers.proposer import UniformMutationProposer, UniformGrowTreeMutationProposer, UniformPruneTreeMutationProposer


class SklearnModel(BaseEstimator, RegressorMixin):

    def __init__(self,
                 n_trees: int=50,
                 sigma_a: int=100.,
                 sigma_b: float=0.001,
                 n_samples: int=200,
                 n_burn: int=200,
                 p_grow: float=0.5,
                 p_prune: float=0.5,
                 alpha: float=0.95,
                 beta: float=2.):
        self.n_trees = n_trees
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.sigma = Sigma(self.sigma_a, self.sigma_b)
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.alpha = alpha
        self.beta = beta
        self.data, self.model, self.proposer, self.sampler, self.prediction_samples, self.model_samples, self.schedule = [None] * 7

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SklearnModel':
        self.data = Data(X, y, normalize=True)
        self.model = Model(self.data, self.sigma, n_trees=self.n_trees, alpha=self.alpha, beta=self.beta)
        self.proposer = UniformMutationProposer({UniformGrowTreeMutationProposer: self.p_grow, UniformPruneTreeMutationProposer: self.p_prune})
        self.schedule = SampleSchedule(self.model, self.proposer)
        self.sampler = Sampler(self.model, self.schedule)
        self.model_samples, self.prediction_samples = self.sampler.samples(self.n_samples, self.n_burn)
        return self

    def predict(self, X: np.ndarray=None):
        if X is None or X == self.data.X:
            return self.data.unnormalize_y(self.prediction_samples.mean(axis=0))
        else:
            return self.out_of_sample_predict(X)

    def out_of_sample_predict(self, X):
        return self.data.unnormalize_y(np.mean([x.out_of_sample_predict(X) for x in self.model_samples], axis=0))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict()
