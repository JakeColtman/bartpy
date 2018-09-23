from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator

from bartpy.model import Model
from bartpy.data import Data
from bartpy.samplers.schedule import SampleSchedule
from bartpy.samplers.modelsampler import ModelSampler
from bartpy.sigma import Sigma
from bartpy.samplers.treemutation.uniform.likihoodratio import UniformTreeMutationLikihoodRatio
from bartpy.samplers.treemutation.uniform.proposer import UniformMutationProposer
from bartpy.samplers.treemutation.treemutation import TreeMutationSampler
from bartpy.samplers.sigma import SigmaSampler
from bartpy.samplers.leafnode import LeafNodeSampler


class SklearnModel(BaseEstimator, RegressorMixin):
    """
    The main access point to building BART models in BartPy

    Parameters
    ----------
    n_trees: int
        the number of trees to use, more trees will make a smoother fit, but slow training and fitting
    sigma_a: float
        shape parameter of the prior on sigma
    sigma_b: float
        scale parameter of the prior on sigma
    n_samples: int
        how many recorded samples to take
    n_burn: int
        how many samples to run without recording to reach convergence
    p_grow: float
        probability of choosing a grow mutation in tree mutation sampling
    p_prune: float
        probability of choosing a prune mutation in tree mutation sampling
    alpha: float
        prior parameter on tree structure
    beta: float
        prior parameter on tree structure
    """

    def __init__(self,
                 n_trees: int=50,
                 sigma_a: float=0.001,
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
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.alpha = alpha
        self.beta = beta
        self.sigma, self.data, self.model, self.proposer, self.likihood_ratio, self.sampler, self._prediction_samples, self._model_samples, self.schedule = [None] * 9

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'SklearnModel':
        """
        Learn the model based on training data

        Parameters
        ----------
        X: pd.DataFrame
            training covariates
        y: np.ndarray
            training targets

        Returns
        -------
        SklearnModel
            self with trained parameter values
        """
        from copy import deepcopy
        self.data = Data(deepcopy(X), deepcopy(y), normalize=True)
        self.sigma = Sigma(self.sigma_a, self.sigma_b, self.data.normalizing_scale)
        self.model = Model(self.data, self.sigma, n_trees=self.n_trees, alpha=self.alpha, beta=self.beta)
        self.proposer = UniformMutationProposer([self.p_grow, self.p_prune])
        self.likihood_ratio = UniformTreeMutationLikihoodRatio([self.p_grow, self.p_prune])
        self.tree_sampler = TreeMutationSampler(self.proposer, self.likihood_ratio)
        self.schedule = SampleSchedule(self.tree_sampler, LeafNodeSampler(), SigmaSampler())
        self.sampler = ModelSampler(self.schedule)
        self._model_samples, self._prediction_samples = self.sampler.samples(self.model, self.n_samples, self.n_burn)
        return self

    def predict(self, X: np.ndarray=None):
        """
        Predict the target corresponding to the provided covariate matrix
        If X is None, will predict based on training covariates

        Prediction is based on the mean of all samples

        Parameters
        ----------
        X: pd.DataFrame
            covariates to predict from

        Returns
        -------
        np.ndarray
            predictions for the X covariates
        """
        if X is None:
            return self.data.unnormalize_y(self._prediction_samples.mean(axis=0))
        else:
            return self._out_of_sample_predict(X)

    def _out_of_sample_predict(self, X):
        return self.data.unnormalize_y(np.mean([x._out_of_sample_predict(X) for x in self._model_samples], axis=0))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict()

    @property
    def model_samples(self) -> List[Model]:
        """
        Array of the model as it was after each sample.
        Useful for examining for:
          * examining the state of trees, nodes and sigma throughout the sampling
          * out of sample prediction

        Returns None if the model hasn't been fit

        Returns
        -------
        List[Model]
        """
        return self._model_samples

    @property
    def prediction_samples(self):
        """
        Matrix of prediction samples at each point in sampling
        Useful for assessing convergence, calculating point estimates etc.

        Returns
        -------
        np.ndarray
            prediction samples with dimensionality n_samples * n_points
        """
        return self.prediction_samples