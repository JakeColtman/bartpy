from copy import deepcopy
from typing import List, Generator

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.tree import Tree, LeafNode, deep_copy_tree
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

        self._prediction = None

    def initialize_trees(self) -> List[Tree]:
        tree_data = deepcopy(self.data)
        tree_data._y = tree_data.y / self.n_trees
        trees = [Tree([LeafNode(Split(self.data))]) for _ in range(self.n_trees)]
        return trees

    def residuals(self) -> np.ndarray:
        return self.data.y - self.predict()

    def unnormalized_residuals(self) -> np.ndarray:
        return self.data.unnormalized_y - self.data.unnormalize_y(self.predict())

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is not None:
            return self._out_of_sample_predict(X)
        return np.sum([tree.predict() for tree in self.trees], axis=0)

    def _out_of_sample_predict(self, X: np.ndarray):
        if type(X) == pd.DataFrame:
            X = X.values
        return np.sum([tree.predict(X) for tree in self.trees], axis=0)

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        if self._prediction is None:
            self._prediction = self.predict()
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


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model
