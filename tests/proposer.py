import unittest

from bartpy.data import Data
from bartpy.proposer import GrowTreeMutationProposer, PruneTreeMutationProposer, ChangeTreeMutationProposer
from bartpy.tree import LeafNode, TreeStructure, SplitNode

import pandas as pd


class TestPruneTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 1]))
        self.d = LeafNode(self.data)
        self.e = LeafNode(self.data)
        self.c = SplitNode(self.data, None, self.d, self.e)
        self.b = LeafNode(self.data)
        self.a = SplitNode(self.data, None, self.b, self.c)
        self.tree_structure = TreeStructure(self.a)
        self.proposer = PruneTreeMutationProposer(self.tree_structure)

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.proposal()
        self.assertIn(proposal.existing_node, self.tree_structure.nodes())
        self.assertNotIn(proposal.updated_node, self.tree_structure.nodes())

    def test_types(self):
        proposal = self.proposer.proposal()
        self.assertIsInstance(proposal.existing_node, SplitNode)
        self.assertIsInstance(proposal.updated_node, LeafNode)


class TestGrowTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 1]))
        self.d = LeafNode(self.data)
        self.e = LeafNode(self.data)
        self.c = SplitNode(self.data, None, self.d, self.e)
        self.b = LeafNode(self.data)
        self.a = SplitNode(self.data, None, self.b, self.c)
        self.tree_structure = TreeStructure(self.a)
        self.proposer = GrowTreeMutationProposer(self.tree_structure)

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.proposal()
        self.assertIn(proposal.existing_node, self.tree_structure.nodes())
        self.assertNotIn(proposal.updated_node, self.tree_structure.nodes())

    def test_types(self):
        proposal = self.proposer.proposal()
        self.assertIsInstance(proposal.updated_node, SplitNode)
        self.assertIsInstance(proposal.updated_node.left_child, LeafNode)
        self.assertIsInstance(proposal.updated_node.right_child, LeafNode)
        self.assertIsInstance(proposal.existing_node, LeafNode)


if __name__ == '__main__':
    unittest.main()
