from bartpy.tree import TreeStructure, TreeNode, LeafNode, SplitNode, TreeMutation

from copy import deepcopy
from unittest import TestCase


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