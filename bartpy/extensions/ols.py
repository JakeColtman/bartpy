from typing import Type

import numpy as np

from bartpy.sklearnmodel import SklearnModel


class OLS(SklearnModel):

    def __init__(self,
                 stat_model: Type,
                 n_trees: int = 50,
                 n_chains: int = 4,
                 sigma_a: float = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 thin: float = 0.1,
                 p_grow: float = 0.5,
                 p_prune: float = 0.5,
                 alpha: float = 0.95,
                 beta: float = 2.,
                 store_in_sample_predictions: bool=True,
                 store_acceptance_trace: bool=True,
                 n_jobs=-1):
        self.stat_model = stat_model
        self.stat_model_fit = None
        super().__init__(n_trees,
                         n_chains,
                         sigma_a,
                         sigma_b,
                         n_samples,
                         n_burn,
                         thin,
                         p_grow,
                         p_prune,
                         alpha,
                         beta,
                         store_in_sample_predictions,
                         store_acceptance_trace,
                         n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLS':
        self.stat_model_fit = self.stat_model(y, X).fit()
        print(self.stat_model_fit.resid)
        SklearnModel.fit(self, X, self.stat_model_fit.resid)
        return self

    def predict(self, X: np.ndarray=None):
        if X is None:
            X = self.data.X
        sm_prediction = self.stat_model_fit.predict(X)
        bart_prediction = SklearnModel.predict(self, X)
        return sm_prediction + bart_prediction
