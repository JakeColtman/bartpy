from copy import deepcopy
from typing import List, Generator

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.tree import Tree, LeafNode
from bartpy.split import Split


class Model:

    def __init__(self, data: Data, sigma: Sigma, trees=None, n_trees: int = 50, alpha: float=0.95, beta: int=2., k: int=2.):
        self.data = data
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma

        if trees is None:
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
        else:
            self.n_trees = len(trees)
            self._trees = trees

        self._prediction = self.predict()

    def initialize_trees(self) -> List[Tree]:
        tree_data = deepcopy(self.data)
        tree_data._y = tree_data.y / self.n_trees
        trees = [Tree([LeafNode(Split(self.data, []))]) for _ in range(self.n_trees)]
        return trees

    def residuals(self) -> pd.Series:
        return self.data.y - self.predict()

    def residuals_without_tree(self, index: int) -> np.ndarray:
        return self.data.y - self.prediction_without_tree(index)

    def predict(self) -> np.ndarray:
        return np.sum([tree.predict() for tree in self.trees], axis=0)

    def prediction_without_tree(self, index: int) -> np.ndarray:
        return np.sum(np.array([tree.predict() for ii, tree in enumerate(self.trees) if ii != index]), axis=0)

    def out_of_sample_predict(self, X: np.ndarray):
        return np.sum([tree.out_of_sample_predict(X) for tree in self.trees], axis=0)

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:

        for tree in self.trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y - self._prediction)
            yield tree
            self._prediction += tree.predict()

    @property
    def sigma_m(self):
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self):
        return self._sigma
