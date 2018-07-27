from typing import List

import numpy as np

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.tree import sample_tree_structure, TreeStructure


class Model:

    def __init__(self, data: Data, sigma: Sigma, n_trees: int = 50, alpha: float=0.95, beta: int=2, k: int=2):
        self.data = data
        self.n_trees = n_trees
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._sigma = sigma
        self._trees = [sample_tree_structure(self.data) for _ in range(self.n_trees)]

    def residuals_tree(self, index: int) -> np.ndarray:
        return self._trees[index].residuals()

    def residuals(self) -> np.ndarray:
        tree_residuals = [tree.residuals() for tree in self.trees]
        return np.sum(tree_residuals, axis=0)

    def residuals_without_tree(self, index: int) -> np.ndarray:
        return self.residuals() - self.residuals_tree(index)

    def predict(self) -> np.ndarray:
        return np.zeros_like(self.data.y)

    @property
    def trees(self) -> List[TreeStructure]:
        return self._trees

    @property
    def sigma_m(self):
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self):
        return self._sigma
