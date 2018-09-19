import numpy as np
import pandas as pd
import statsmodels.api as sm

from bartpy.sklearnmodel import SklearnModel


class OLS(SklearnModel):

    def __init__(self,
                 stat_model: sm.OLS,
                 n_trees: int = 50,
                 sigma_a: int = 0.001,
                 sigma_b: float = 0.001,
                 n_samples: int = 200,
                 n_burn: int = 200,
                 p_grow: float = 0.5,
                 p_prune: float = 0.5,
                 alpha: float = 0.95,
                 beta: float = 2.):
        self.stat_model = stat_model
        self.stat_model_fit = None
        super().__init__(n_trees, sigma_a, sigma_b, n_samples, n_burn, p_grow, p_prune, alpha, beta)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'OLS':
        self.stat_model_fit = self.stat_model(y, X).fit()
        print(self.stat_model_fit.resid)
        SklearnModel.fit(self, X, self.stat_model_fit.resid)
        return self
