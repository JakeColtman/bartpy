from copy import deepcopy, copy
from operator import gt, le
from typing import List, Generator

import numpy as np
import pandas as pd

from bartpy.data import Data
from bartpy.mutation import GrowMutation
from bartpy.node import split_node
from bartpy.sigma import Sigma
from bartpy.split import Split
from bartpy.splitcondition import SplitCondition
from bartpy.tree import Tree, LeafNode, deep_copy_tree, mutate


class Model:

    def __init__(self,
                 data: Data,
                 sigma: Sigma,
                 trees=None,
                 n_trees: int = 50,
                 alpha: float=0.95,
                 beta: float=2.,
                 k: int=2.):

        self.data = data
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k = k
        self._sigma = sigma
        self._prediction = None

        if trees is None:
            self.n_trees = n_trees
            self._trees = self.initialize_trees()
            self.initialize_tree_structure()
        else:
            self.n_trees = len(trees)
            self._trees = trees

    def initialize_trees(self) -> List[Tree]:
        tree_data = copy(self.data)
        tree_data.update_y(tree_data.y / self.n_trees)
        trees = [Tree([LeafNode(Split(tree_data))]) for _ in range(self.n_trees)]
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
            X: pd.DataFrame = X
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

    def initialize_tree_structure(self):
        from sklearn.ensemble import GradientBoostingRegressor
        for tree in self.refreshed_trees():
            params = {'n_estimators': 1, 'max_depth': 4, 'min_samples_split': 2,
                      'learning_rate': 0.8, 'loss': 'ls'}
            clf = GradientBoostingRegressor(**params)
            fit = clf.fit(tree.nodes[0].data.X.data, tree.nodes[0].data.y.data)
            sklearn_tree = fit.estimators_[0][0].tree_
            map_sklearn_tree_into_bartpy(tree, sklearn_tree)


def map_sklearn_split_into_bartpy_split_conditons(tree, index):
    return [SplitCondition(tree.feature[index], tree.threshold[index], le), SplitCondition(tree.feature[index], tree.threshold[index], gt)]


def map_sklearn_tree_into_bartpy(bartpy_tree, sklearn_tree):
    nodes = [None for x in sklearn_tree.children_left]
    nodes[0] = bartpy_tree.nodes[0]

    def search(bartpy_tree, sklearn_tree, parent_index=0, index=0):
        if sklearn_tree.children_left[index] == -1:
            return

        split_conditions = map_sklearn_split_into_bartpy_split_conditons(sklearn_tree, index)
        decision_node = split_node(nodes[index], split_conditions)
        decision_node.left_child.set_value(sklearn_tree.value[sklearn_tree.children_left[index]][0][0])
        decision_node.right_child.set_value(sklearn_tree.value[sklearn_tree.children_right[index]][0][0])

        mutation = GrowMutation(nodes[index], decision_node)
        mutate(bartpy_tree, mutation)
        nodes[index] = decision_node
        nodes[sklearn_tree.children_left[index]] = decision_node.left_child
        nodes[sklearn_tree.children_right[index]] = decision_node.right_child

        search(bartpy_tree, sklearn_tree, index, sklearn_tree.children_left[index])
        search(bartpy_tree, sklearn_tree, index, sklearn_tree.children_right[index])

    search(bartpy_tree, sklearn_tree)


def deep_copy_model(model: Model) -> Model:
    copied_model = Model(None, deepcopy(model.sigma), [deep_copy_tree(tree) for tree in model.trees])
    return copied_model

