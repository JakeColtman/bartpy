import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data, format_covariate_matrix
from bartpy.mutation import GrowMutation, PruneMutation
from bartpy.node import DecisionNode, LeafNode
from bartpy.split import Split


class TestMutation(unittest.TestCase):

    def setUp(self):
        self.X = format_covariate_matrix(pd.DataFrame({"a": [1]}))
        self.data = Data(format_covariate_matrix(self.X), np.array([1.0]))

    def test_pruning_leaf(self):
        with self.assertRaises(TypeError):
            PruneMutation(LeafNode(Split(self.data)), LeafNode(Split(self.data)))

    def test_growing_decision_node(self):
        a = LeafNode(Split(self.data))
        b = LeafNode(Split(self.data))
        c = LeafNode(Split(self.data))
        d = DecisionNode(Split(self.data), a, b)
        e = DecisionNode(Split(self.data), c, d)

        with self.assertRaises(TypeError):
            GrowMutation(d, a)

    def test_pruning_non_leaf_parent(self):
        a = LeafNode(Split(self.data))
        b = LeafNode(Split(self.data))
        c = LeafNode(Split(self.data))
        d = DecisionNode(Split(self.data), a, b)
        e = DecisionNode(Split(self.data), c, d)

        with self.assertRaises(TypeError):
            PruneMutation(e, a)

if __name__ == '__main__':
    unittest.main()
