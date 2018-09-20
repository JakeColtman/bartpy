import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model.base import LinearRegression

from bartpy.sklearnmodel import SklearnModel


class ResidualBART(SklearnModel):

    def __init__(self,
                 base_estimator: RegressorMixin = None,
                 n_trees: int = 50,
                 sigma_a: int = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 p_grow: float = 0.5,
                 p_prune: float = 0.5,
                 alpha: float = 0.95,
                 beta: float = 2.):

        if base_estimator is not None:
            self.base_estimator = clone(base_estimator)
        else:
            base_estimator = LinearRegression()
        self.base_estimator = base_estimator
        super().__init__(n_trees, sigma_a, sigma_b, n_samples, n_burn, p_grow, p_prune, alpha, beta)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'ResidualBART':
        self.base_estimator.fit(X, y)
        SklearnModel.fit(self, X, y - self.base_estimator.predict(X))
        return self

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is None:
            X = self.data.X
        sm_prediction = self.base_estimator.predict(X)
        bart_prediction = SklearnModel.predict(self, X)
        return sm_prediction + bart_prediction
