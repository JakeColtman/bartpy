import unittest
from unittest import TestCase
from operator import le, gt

import pandas as pd
import numpy as np

from bartpy.data import Data, format_covariate_matrix, make_bartpy_data
from bartpy.initializers.sklearntreeinitializer import map_sklearn_tree_into_bartpy
from bartpy.mutation import TreeMutation, PruneMutation
from bartpy.node import split_node, LeafNode, DecisionNode
from bartpy.tree import mutate, Tree
from bartpy.split import Split, SplitCondition




class TestTreeStructureNodeRetrieval(TestCase):

    def setUp(self):
        X = format_covariate_matrix(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}))
        data = Data(X, np.array([1, 2, 3]).astype(float))
        split = Split(data)
        node = LeafNode(split)
        self.a = split_node(node, (SplitCondition(0, 1, le), SplitCondition(0, 1, gt)))
        self.tree = Tree([self.a, self.a.left_child, self.a.right_child])

        self.c = split_node(self.a._right_child, (SplitCondition(1, 2, le), SplitCondition(1, 2, gt)))
        mutate(self.tree, TreeMutation("grow", self.a.right_child, self.c))

        self.b = self.a.left_child
        self.d = self.c.left_child
        self.e = self.c.right_child

    def test_retrieve_all_nodes(self):
        all_nodes = self.tree.nodes
        for node in [self.a, self.b, self.c, self.d, self.e]:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, [self.a, self.b, self.c, self.d, self.e])

    def test_retrieve_all_leaf_nodes(self):
        all_nodes = self.tree.leaf_nodes
        true_all_nodes = [self.d, self.e, self.b]
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)

    def test_retrieve_all_leaf_parents(self):
        all_nodes = self.tree.prunable_decision_nodes
        true_all_nodes = [self.c]
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)

    def test_retrieve_all_split_nodes(self):
        all_nodes = self.tree.decision_nodes
        true_all_nodes = [self.c, self.a]
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)


class TestTreeStructureDataUpdate(TestCase):

    def setUp(self):
        X = format_covariate_matrix(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}))
        self.data = Data(X, np.array([1, 2, 3]).astype(float))

        self.a = split_node(LeafNode(Split(self.data)), (SplitCondition(0, 1, le), SplitCondition(0, 1, gt)))
        self.b = self.a.left_child
        self.x = self.a.right_child
        self.tree = Tree([self.a, self.b, self.x])

        self.c = split_node(self.a._right_child, (SplitCondition(1, 2, le), SplitCondition(1, 2, gt)))
        mutate(self.tree, TreeMutation("grow", self.x, self.c))

        self.d = self.c.left_child
        self.e = self.c.right_child

    def test_update_pushed_through_split(self):
        updated_y = np.array([5, 6, 7])
        self.tree.update_y(updated_y)
        # Left child keeps LTE condition
        self.assertListEqual([5, 6, 7], list(self.a.data.y.values))
        self.assertListEqual([5], list(self.b.data.y.values[~self.b.data.y._mask]))
        self.assertListEqual([6, 7], list(self.c.data.y.values[~self.c.data.y._mask]))
        self.assertListEqual([6], list(self.d.data.y.values[~self.d.data.y._mask]))
        self.assertListEqual([7], list(self.e.data.y.values[~self.e.data.y._mask]))


class TestTreeStructureMutation(TestCase):

    def setUp(self):
        self.data = Data(format_covariate_matrix(pd.DataFrame({"a": [1]})), np.array([1]).astype(float))
        self.d = LeafNode(Split(self.data), None)
        self.e = LeafNode(Split(self.data), None)
        self.c = DecisionNode(Split(self.data), self.d, self.e)
        self.b = LeafNode(Split(self.data))
        self.a = DecisionNode(Split(self.data), self.b, self.c)
        self.tree = Tree([self.a, self.b, self.c, self.d, self.e])

    def test_starts_right(self):
        self.assertListEqual([self.c], self.tree.prunable_decision_nodes)
        for leaf in [self.b, self.d, self.e]:
            self.assertIn(leaf, self.tree.leaf_nodes)

    def test_invalid_prune(self):
        with self.assertRaises(TypeError):
            updated_a = LeafNode(Split(self.data))
            PruneMutation(self.a, updated_a)

    def test_grow(self):
        f, g = LeafNode(Split(self.data)), LeafNode(Split(self.data))
        updated_d = DecisionNode(Split(self.data), f, g)
        grow_mutation = TreeMutation("grow", self.d, updated_d)
        mutate(self.tree, grow_mutation)
        self.assertIn(updated_d, self.tree.decision_nodes)
        self.assertIn(updated_d, self.tree.prunable_decision_nodes)
        self.assertIn(f, self.tree.leaf_nodes)
        self.assertNotIn(self.d, self.tree.nodes)

    def test_head_prune(self):
        b, c = LeafNode(Split(self.data)), LeafNode(Split(self.data))
        a = DecisionNode(Split(self.data), b, c)
        tree = Tree([a, b, c])
        updated_a = LeafNode(Split(self.data))
        prune_mutation = PruneMutation(a, updated_a)
        mutate(tree, prune_mutation)
        self.assertIn(updated_a, tree.leaf_nodes)
        self.assertNotIn(self.a, tree.nodes)

    def test_internal_prune(self):
        updated_c = LeafNode(Split(self.data))
        prune_mutation = TreeMutation("prune", self.c, updated_c)
        mutate(self.tree, prune_mutation)
        self.assertIn(updated_c, self.tree.leaf_nodes)
        self.assertNotIn(self.c, self.tree.nodes)
        self.assertNotIn(self.d, self.tree.nodes)
        self.assertNotIn(self.e, self.tree.nodes)


class TestSklearnToBartPyTreeMapping(unittest.TestCase):

    def setUp(self):
        self.X = np.random.normal(size=20)
        self.y = self.X + np.random.normal(scale=0.1, size=20)
        self.data = make_bartpy_data(pd.DataFrame({"a": self.X}), self.y, normalize=False)

    def test_same_prediction(self):
        from sklearn.ensemble import GradientBoostingRegressor
        params = {'n_estimators': 1, 'max_depth': 2, 'min_samples_split': 2,
                  'learning_rate': 0.8, 'loss': 'ls'}
        sklearn_model = GradientBoostingRegressor(**params)
        sklearn_model.fit(self.data.X.values, self.data.y.values)

        sklearn_tree = sklearn_model.estimators_[0][0].tree_
        bartpy_tree = Tree([LeafNode(Split(self.data))])

        map_sklearn_tree_into_bartpy(bartpy_tree, sklearn_tree)

        sklearn_predictions = sklearn_tree.predict(self.data.X.values.astype(np.float32))
        sklearn_predictions = [round(x, 2) for x in sklearn_predictions.reshape(-1)]

        bartpy_tree.cache_up_to_date = False
        bartpy_tree_predictions = bartpy_tree.predict(self.data.X.values)
        bartpy_tree_predictions = [round(x, 2) for x in bartpy_tree_predictions]

        self.assertListEqual(sklearn_predictions, bartpy_tree_predictions)


if __name__ == '__main__':
    unittest.main()
