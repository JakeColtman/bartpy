from operator import gt, le
import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data, format_covariate_matrix
from bartpy.mutation import GrowMutation, PruneMutation
from bartpy.node import DecisionNode, LeafNode, split_node
from bartpy.split import Split
from bartpy.splitcondition import SplitCondition

class TestNode(unittest.TestCase):

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


class TestSplitNode(unittest.TestCase):

    def setUp(self):
        self.X = format_covariate_matrix(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
        self.data = Data(format_covariate_matrix(self.X), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.split = Split(self.data)
        self.node = LeafNode(self.split)
    
    def test_unsplit(self):
        self.assertEqual(self.node.data.y.summed_y(), 15.)

    def test_split(self):
        left_split_condition = SplitCondition(0, 3, le)
        right_split_condition = SplitCondition(0, 3, gt)
        updated_node = split_node(self.node, [left_split_condition, right_split_condition])
        self.assertIsInstance(updated_node, DecisionNode)

        self.assertEqual(updated_node.data.y.summed_y(), 15)
        self.assertEqual(updated_node.left_child.data.y.summed_y(), 6)
        self.assertEqual(updated_node.right_child.data.y.summed_y(), 9)

        self.assertEqual(updated_node.data.X.n_obsv, 5)
        self.assertEqual(updated_node.left_child.data.X.n_obsv, 3)
        self.assertEqual(updated_node.right_child.data.X.n_obsv, 2)

        updated_node.update_y([2.0, 4.0, 6.0, 8.0, 10.0])

        self.assertEqual(updated_node.left_child.data.y.summed_y(), 12)


if __name__ == '__main__':
    unittest.main()
