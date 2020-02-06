from copy import deepcopy, copy
from typing import List, Generator, Optional

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.initializers.initializer import Initializer
from bartpy.initializers.sklearntreeinitializer import SklearnTreeInitializer
from bartpy.sigma import Sigma
from bartpy.split import Split
from bartpy.tree import Tree, LeafNode, deep_copy_tree


class Model:

    def __init__(self,
                 data: Optional[Data],
                 sigma: Sigma,
                 trees: Optional[List[Tree]]=None,
                 n_trees: int=50,
                 alpha: float=0.95,
                 beta: float=2.,
                 k: int=2.,
                 initializer: Initializer=SklearnTreeInitializer()):

        self.data = deepcopy(data)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None
        self._initializer = initializer

        if trees is None:
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            if self._initializer is not None:
                self._initializer.initialize_trees(self.refreshed_trees())
        else:
            self.n_trees = len(trees)
            self._trees = trees

    def initialize_trees(self) -> List[Tree]:
        trees = [Tree([LeafNode(Split(deepcopy(self.data)))]) for _ in range(self.n_trees)]
        for tree in trees:
            tree.update_y(tree.update_y(self.data.y.values / self.n_trees))
        return trees

    def residuals(self) -> np.ndarray:
        return self.data.y.values - self.predict()

    def unnormalized_residuals(self) -> np.ndarray:
        return self.data.y.unnormalized_y - self.data.y.unnormalize_y(self.predict())

    def predict(self, X: np.ndarray=None) -> np.ndarray:
        if X is not None:
            return self._out_of_sample_predict(X)
        return np.sum([tree.predict() for tree in self.trees], axis=0)

    def _out_of_sample_predict(self, X: np.ndarray) -> np.ndarray:
        if type(X) == pd.DataFrame:
            X: pd.DataFrame = X
            X = X.values
        return np.sum([tree.predict(X) for tree in self.trees], axis=0)

    @property
    def trees(self) -> List[Tree]:
        return self._trees

    def refreshed_trees(self) -> Generator[Tree, None, None]:
        if self._prediction is None:
            self._prediction = self.predict()
        for tree in self._trees:
            self._prediction -= tree.predict()
            tree.update_y(self.data.y.values - self._prediction)
            yield tree
            self._prediction += tree.predict()

    @property
    def sigma_m(self) -> float:
        return 0.5 / (self.k * np.power(self.n_trees, 0.5))

    @property
    def sigma(self) -> Sigma:
        return self._sigma


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model
