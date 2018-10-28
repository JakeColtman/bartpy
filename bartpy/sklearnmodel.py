from typing import Union, List

from joblib import Parallel, delayed
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
    n_chains: int
        the number of independent chains to run
        more chains will improve the quality of the samples, but will require more computation
    sigma_a: float
        shape parameter of the prior on sigma
    sigma_b: float
        scale parameter of the prior on sigma
    n_samples: int
        how many recorded samples to take
    n_burn: int
        how many samples to run without recording to reach convergence
    thin: float
        percentage of samples to store.
        use this to save memory when running large models
    p_grow: float
        probability of choosing a grow mutation in tree mutation sampling
    p_prune: float
        probability of choosing a prune mutation in tree mutation sampling
    alpha: float
        prior parameter on tree structure
    beta: float
        prior parameter on tree structure
    store_in_sample_predictions: bool
        whether to store full prediction samples
        set to False if you don't need in sample results - saves a lot of memory
    n_jobs: int
        how many cores to use when computing MCMC samples
    """

    def __init__(self,
                 n_trees: int=50,
                 n_chains: int=4,
                 sigma_a: float=0.001,
                 sigma_b: float=0.001,
                 n_samples: int=200,
                 n_burn: int=200,
                 thin: float=0.1,
                 p_grow: float=0.5,
                 p_prune: float=0.5,
                 alpha: float=0.95,
                 beta: float=2.,
                 store_in_sample_predictions: bool=True,
                 n_jobs=4):
        self.n_trees = n_trees
        self.n_chains = n_chains
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.alpha = alpha
        self.beta = beta
        self.thin = thin
        self.n_jobs = n_jobs
        self.store_in_sample_predictions = store_in_sample_predictions
        self.columns = None

        self.proposer = UniformMutationProposer([self.p_grow, self.p_prune])
        self.likihood_ratio = UniformTreeMutationLikihoodRatio([self.p_grow, self.p_prune])
        self.tree_sampler = TreeMutationSampler(self.proposer, self.likihood_ratio)
        self.schedule = SampleSchedule(self.tree_sampler, LeafNodeSampler(), SigmaSampler())
        self.sampler = ModelSampler(self.schedule)

        self.sigma, self.data, self.model, self._prediction_samples, self._model_samples, self.extract = [None] * 6

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> 'SklearnModel':
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
        self.extract = Parallel(n_jobs=self.n_jobs)(self.delayed_chains(X, y))
        self._model_samples, self._prediction_samples = self._combine_chains(self.extract)
        return self

    @staticmethod
    def _combine_chains(extract):
        model_samples, prediction_samples = extract[0]
        for x in extract[1:]:
            model_samples += x[0]
            prediction_samples = np.concatenate([prediction_samples, x[1]], axis=0)
        return model_samples, prediction_samples

    def _convert_covariates_to_data(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> Data:
        from copy import deepcopy
        if type(X) == pd.DataFrame:
            self.columns = X.columns
            X = X.values
        else:
            self.columns = list(map(str, range(X.shape[1])))

        return Data(deepcopy(X), deepcopy(y), normalize=True)

    def _construct_model(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> Model:
        self.data = self._convert_covariates_to_data(X, y)
        self.sigma = Sigma(self.sigma_a, self.sigma_b, self.data.normalizing_scale)
        self.model = Model(self.data, self.sigma, n_trees=self.n_trees, alpha=self.alpha, beta=self.beta)
        return self.model

    def delayed_chains(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        self.model = self._construct_model(X, y)
        return [delayed(self.sampler.samples)(self.model,
                                              self.n_samples,
                                              self.n_burn,
                                              self.thin,
                                              self.store_in_sample_predictions) for x in range(self.n_chains)]

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

    def residuals(self, X=None):
        return self.model.data.unnormalized_y - self.predict(X)

    def l2_error(self, X=None):
        return np.square(self.residuals(X))

    def _out_of_sample_predict(self, X):
        return self.data.unnormalize_y(np.mean([x.predict(X) for x in self._model_samples], axis=0))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict()

    @property
    def model_samples(self) -> List[Model]:
        """
        Array of the model as it was after each sample.
        Useful for examining for:

         - examining the state of trees, nodes and sigma throughout the sampling
         - out of sample prediction

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