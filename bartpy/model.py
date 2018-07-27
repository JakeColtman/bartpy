from copy import deepcopy
from typing import List, Generator

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.tree import sample_tree_structure, TreeStructure


class Model:

    def __init__(self, data: Data, sigma: Sigma, n_trees: int = 50, alpha: float=0.95, beta: int=2., k: int=2.):
        self.data = data
        self.n_trees = n_trees
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._trees = self.initialize_trees()

    def initialize_trees(self) -> List[TreeStructure]:
        tree_data = deepcopy(self.data)
        tree_data._y = tree_data.y / self.n_trees
        trees = [sample_tree_structure(tree_data, self.alpha, self.beta) for _ in range(self.n_trees)]
        return trees

    def residuals(self) -> pd.Series:
        return self.data.y - self.predict()

    def residuals_without_tree(self, index: int) -> np.ndarray:
        return self.residuals() + self.prediction_without_tree(index)

    def predict(self) -> pd.Series:
        return pd.Series(np.sum([tree.predict(self.data) for tree in self.trees], axis=0))

    def prediction_without_tree(self, index: int) -> pd.Series:
        full_prediction = self.predict()
        tree_prediction = self.trees[index].predict(self.data)
        return full_prediction - tree_prediction

    @property
    def trees(self) -> List[TreeStructure]:
        return self._trees

    def refreshed_trees(self) -> Generator[TreeStructure, None, None]:
        for index, tree in enumerate(self.trees):
            tree_data = deepcopy(self.data)
            tree_data._y = self.residuals_without_tree(index)
            tree.update_data(tree_data)
            yield tree

    @property
    def sigma_m(self):
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self):
        return self._sigma


if __name__ == "__main__":
    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma)
    full_prediction = model.predict()
    tree_prediction = model.trees[1].predict(data)

    #print(model.residuals())
    print(model.residuals_without_tree(0))
    #print(tree_prediction)
    #print(full_prediction)