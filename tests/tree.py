from unittest import TestCase

from bartpy.tree import TreeStructure, TreeNode, LeafNode, SplitNode, TreeMutation
from bartpy.data import Data, Split

import pandas as pd


class TestTreeStructureDataUpdate(TestCase):

    def setUp(self):
        self.d = LeafNode(None)
        self.e = LeafNode(None)
        self.c = SplitNode(None, None, self.d, self.e)
        self.b = LeafNode(None)
        self.a = SplitNode(None, None, self.b, self.c)
        self.tree_structure = TreeStructure(self.a)

    def test_leaf_node_data_update(self):
        self.assertIsNone(self.b.data)
        self.b.update_data(10)
        self.assertEqual(self.b.data, 10)

    def test_update_pushed_through_split(self):
        data_pd = pd.DataFrame({"a": [1, 2]})
        updated_data = Data(data_pd, pd.Series([0, 1]))
        split = Split("a", 1)
        self.c = SplitNode(None, split, self.d, self.e)
        self.c.update_data(updated_data)
        # Split node keeps updated copy of data
        self.assertListEqual([1, 2], list(self.c.data.X["a"]))
        # Left child keeps LTE condition
        self.assertListEqual([1], list(self.c.left_child.data.X["a"]))
        # Right child keeps GT condition
        self.assertListEqual([2], list(self.c.right_child.data.X["a"]))
        self.assertListEqual([1], list(self.c.right_child.data.y))


class TestTreeStructureMutation(TestCase):

    def setUp(self):
        self.d = LeafNode(None)
        self.e = LeafNode(None)
        self.c = SplitNode(None, None, self.d, self.e)
        self.b = LeafNode(None)
        self.a = SplitNode(None, None, self.b, self.c)
        self.tree_structure = TreeStructure(self.a)

    def test_grow(self):
        f, g = LeafNode(None), LeafNode(None)
        updated_d = SplitNode(None, None, f, g)
        grow_mutation = TreeMutation("grow", self.d, updated_d)
        self.tree_structure.update_node(grow_mutation)
        self.assertEqual(self.tree_structure.head.right_child.left_child, updated_d)
        self.assertIsNone(self.tree_structure.head.right_child.left_child.left_child.left_child)

    def test_head_prune(self):
        updated_a = LeafNode(None)
        prune_mutation = TreeMutation("prune", self.a, updated_a)
        self.tree_structure.update_node(prune_mutation)
        self.assertEqual(self.tree_structure.head, updated_a)
        self.assertIsNone(self.tree_structure.head.left_child)

    def test_internal_prune(self):
        updated_c = LeafNode(None)
        prune_mutation = TreeMutation("prune", self.c, updated_c)
        self.tree_structure.update_node(prune_mutation)
        self.assertEqual(self.tree_structure.head.right_child, updated_c)
        self.assertIsNone(self.tree_structure.head.right_child.left_child)
