import unittest

from bartpy.data import make_bartpy_data
from bartpy.samplers.unconstrainedtree.proposer import uniformly_sample_grow_mutation, uniformly_sample_prune_mutation
from bartpy.split import Split
from bartpy.tree import LeafNode, Tree, DecisionNode

import pandas as pd
import numpy as np


class TestPruneTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = make_bartpy_data(pd.DataFrame({"a": [1, 2]}), np.array([1, 2]), normalize=False)
        self.d = LeafNode(Split(self.data))
        self.e = LeafNode(Split(self.data))
        self.c = DecisionNode(Split(self.data), self.d, self.e)
        self.b = LeafNode(Split(self.data))
        self.a = DecisionNode(Split(self.data), self.b, self.c)
        self.tree = Tree([self.a, self.b, self.c, self.d, self.e])

    def test_proposal_isnt_mutating(self):
        proposal = uniformly_sample_prune_mutation(self.tree)
        self.assertIn(proposal.existing_node, self.tree.nodes)
        self.assertNotIn(proposal.updated_node, self.tree.nodes)

    def test_types(self):
        proposal = uniformly_sample_prune_mutation(self.tree)
        self.assertIsInstance(proposal.existing_node, DecisionNode)
        self.assertIsInstance(proposal.updated_node, LeafNode)


class TestGrowTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = make_bartpy_data(pd.DataFrame({"a": np.random.normal(size=1000)}), np.array(np.random.normal(size=1000)))
        self.d = LeafNode(Split(self.data))
        self.e = LeafNode(Split(self.data))
        self.c = DecisionNode(Split(self.data), self.d, self.e)
        self.b = LeafNode(Split(self.data))
        self.a = DecisionNode(Split(self.data), self.b, self.c)
        self.tree = Tree([self.a, self.b, self.c, self.d, self.e])

    def test_proposal_isnt_mutating(self):
        proposal = uniformly_sample_grow_mutation(self.tree)
        self.assertIn(proposal.existing_node, self.tree.nodes)
        self.assertNotIn(proposal.updated_node, self.tree.nodes)

    def test_types(self):
        proposal = uniformly_sample_grow_mutation(self.tree)
        self.assertIsInstance(proposal.updated_node, DecisionNode)
        self.assertIsInstance(proposal.updated_node.left_child, LeafNode)
        self.assertIsInstance(proposal.updated_node.right_child, LeafNode)
        self.assertIsInstance(proposal.existing_node, LeafNode)


if __name__ == '__main__':
    unittest.main()
