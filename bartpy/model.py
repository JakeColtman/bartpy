from typing import List

import numpy as np

from bartpy.data import Data
from bartpy.tree import sample_tree_structure, TreeStructure


class Model:

    def __init__(self, X: Data, y: np.ndarray, n_trees: int = 50, alpha=0.95, beta=2):
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.alpha = alpha
        self.beta = beta

        self._trees = [sample_tree_structure(self.X) for _ in range(self.n_trees)]

    def residuals_tree(self, index: int) -> np.ndarray:
        return self._trees[index].residuals()

    def residuals(self) -> np.ndarray:
        tree_residuals = [tree.residuals() for tree in self.trees]
        return np.sum(tree_residuals, axis=0)

    def residuals_without_tree(self, index: int) -> np.ndarray:
        return self.residuals() - self.residuals_tree(index)

    @property
    def trees(self) -> List[TreeStructure]:
        return self._trees

