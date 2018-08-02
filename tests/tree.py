from unittest import TestCase

from bartpy.data import Data
from bartpy.tree import TreeStructure, LeafNode, SplitNode, TreeMutation, PruneMutation, split_node
from bartpy.split import Split, SplitCondition, LTESplitCondition
import pandas as pd


class TestTreeStructureNodeRetrieval(TestCase):

    def setUp(self):
        data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series([1, 2, 3]))
        self.a = split_node(LeafNode(data), SplitCondition("a", 1))
        self.tree_structure = TreeStructure(self.a)

        self.c = split_node(self.a._right_child, SplitCondition("b", 1))
        self.tree_structure.mutate(TreeMutation("grow", self.a.right_child, self.c))

        self.b = self.a.left_child
        self.d = self.c.left_child
        self.e = self.c.right_child

    def test_retrieve_all_nodes(self):
        all_nodes = self.tree_structure.nodes()
        for node in [self.a, self.b, self.c, self.d, self.e]:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, [self.a, self.b, self.c, self.d, self.e])

    def test_retrieve_all_leaf_nodes(self):
        all_nodes = self.tree_structure.leaf_nodes()
        true_all_nodes = [self.d, self.e, self.b]
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)

    def test_retrieve_all_leaf_parents(self):
        all_nodes = self.tree_structure.leaf_parents()
        true_all_nodes = [self.c]
        print(self.c.is_leaf_parent())
        print(all_nodes)
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)

    def test_retrieve_all_split_nodes(self):
        all_nodes = self.tree_structure.split_nodes()
        true_all_nodes = [self.c, self.a]
        for node in true_all_nodes:
            self.assertIn(node, all_nodes)
        for node in all_nodes:
            self.assertIn(node, true_all_nodes)


class TestTreeStructureDataUpdate(TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series([1, 2, 3]))
        self.a = split_node(LeafNode(self.data), SplitCondition("a", 1))
        self.tree_structure = TreeStructure(self.a)

        self.c = split_node(self.a._right_child, SplitCondition("b", 2))
        self.tree_structure.mutate(TreeMutation("grow", self.a.right_child, self.c))

        self.b = self.a.left_child
        self.d = self.c.left_child
        self.e = self.c.right_child

    def test_update_pushed_through_split(self):
        updated_y = pd.Series([5, 6, 7])
        self.tree_structure.update_y(updated_y)
        # Left child keeps LTE condition
        self.assertListEqual([5, 6, 7], list(self.a.data.y))
        self.assertListEqual([5], list(self.b.data.y))
        self.assertListEqual([6, 7], list(self.c.data.y))
        self.assertListEqual([6], list(self.d.data.y))
        self.assertListEqual([7], list(self.e.data.y))


class TestTreeStructureMutation(TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1]}), pd.Series([1]))
        self.d = LeafNode(self.data, None)
        self.e = LeafNode(self.data, None)
        self.c = SplitNode(self.data, None, self.d, self.e)
        self.b = LeafNode(self.data, None)
        self.a = SplitNode(self.data, None, self.b, self.c)
        self.tree_structure = TreeStructure(self.a)

    def test_starts_right(self):
        self.assertListEqual([self.c], self.tree_structure.leaf_parents())
        for leaf in [self.b, self.d, self.e]:
            self.assertIn(leaf, self.tree_structure.leaf_nodes())

    def test_invalid_prune(self):
        with self.assertRaises(TypeError):
            updated_a = LeafNode(self.data, None)
            PruneMutation(self.a, updated_a)

    def test_grow(self):
        f, g = LeafNode(self.data, None), LeafNode(self.data, None)
        updated_d = SplitNode(None, None, f, g)
        grow_mutation = TreeMutation("grow", self.d, updated_d)
        self.tree_structure.mutate(grow_mutation)
        self.assertIn(updated_d, self.tree_structure.split_nodes())
        self.assertIn(updated_d, self.tree_structure.leaf_parents())
        self.assertIn(f, self.tree_structure.leaf_nodes())
        self.assertNotIn(self.d, self.tree_structure.nodes())

    def test_head_prune(self):
        a = SplitNode(self.data, None, LeafNode(self.data, None), LeafNode(self.data, None))
        tree_structure = TreeStructure(a)
        updated_a = LeafNode(self.data, None)
        print(a.is_leaf_parent())
        prune_mutation = PruneMutation(a, updated_a)
        tree_structure.mutate(prune_mutation)
        self.assertIn(updated_a, tree_structure.leaf_nodes())
        self.assertNotIn(self.a, tree_structure.nodes())

    def test_internal_prune(self):
        updated_c = LeafNode(self.data, None)
        prune_mutation = TreeMutation("prune", self.c, updated_c)
        self.tree_structure.mutate(prune_mutation)
        self.assertIn(updated_c, self.tree_structure.leaf_nodes())
        self.assertNotIn(self.c, self.tree_structure.nodes())
        self.assertNotIn(self.d, self.tree_structure.nodes())
        self.assertNotIn(self.e, self.tree_structure.nodes())
