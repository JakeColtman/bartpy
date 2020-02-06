from operator import gt, le
import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data, format_covariate_matrix
from bartpy.model import Model
from bartpy.mutation import GrowMutation, PruneMutation
from bartpy.node import DecisionNode, LeafNode, split_node
from bartpy.sigma import Sigma
from bartpy.split import Split
from bartpy.splitcondition import SplitCondition


class TestSplitModel(unittest.TestCase):

    def setUp(self):
        self.X = format_covariate_matrix(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
        self.raw_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.data = Data(format_covariate_matrix(self.X), self.raw_y, normalize=True)
        normalizing_scale = self.data.y.normalizing_scale
        self.model = Model(self.data, Sigma(0.001, 0.001, scaling_factor=normalizing_scale), n_trees=2, initializer=None)
        self.model.initialize_trees()

    def test_tree_updating(self):
        updated_y = np.ones(5)
        self.model.trees[0].update_y(updated_y)
        self.assertListEqual(list(self.model.trees[0].nodes[0].data.y.values), list(updated_y))

    def test_trees_initialized_correctly(self):
        self.assertEqual(len(self.model.trees), 2)
        for tree in self.model.trees:
            self.assertEqual(len(tree.nodes), 1)


if __name__ == '__main__':
    unittest.main()
