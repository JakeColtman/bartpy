import unittest

from bartpy.data import Data
from bartpy.proposer import GrowTreeMutationProposer, PruneTreeMutationProposer
from bartpy.split import Split
from bartpy.tree import LeafNode, Tree, DecisionNode

import pandas as pd


class TestPruneTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 1]))
        self.d = LeafNode(Split(self.data))
        self.e = LeafNode(Split(self.data))
        self.c = DecisionNode(Split(self.data), self.d, self.e)
        self.b = LeafNode(Split(self.data))
        self.a = DecisionNode(Split(self.data), self.b, self.c)
        self.tree_structure = Tree([self.a, self.b, self.c, self.d, self.e])
        self.proposer = PruneTreeMutationProposer(self.tree_structure)

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.proposal()
        self.assertIn(proposal.existing_node, self.tree_structure.nodes)
        self.assertNotIn(proposal.updated_node, self.tree_structure.nodes)

    def test_types(self):
        proposal = self.proposer.proposal()
        self.assertIsInstance(proposal.existing_node, DecisionNode)
        self.assertIsInstance(proposal.updated_node, LeafNode)


class TestGrowTreeMutationProposer(unittest.TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 1]))
        self.d = LeafNode(Split(self.data))
        self.e = LeafNode(Split(self.data))
        self.c = DecisionNode(Split(self.data), self.d, self.e)
        self.b = LeafNode(Split(self.data))
        self.a = DecisionNode(Split(self.data), self.b, self.c)
        self.tree_structure = Tree([self.a, self.b, self.c, self.d, self.e])
        self.proposer = GrowTreeMutationProposer(self.tree_structure)

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.proposal()
        self.assertIn(proposal.existing_node, self.tree_structure.nodes)
        self.assertNotIn(proposal.updated_node, self.tree_structure.nodes)

    def test_types(self):
        proposal = self.proposer.proposal()
        self.assertIsInstance(proposal.updated_node, DecisionNode)
        self.assertIsInstance(proposal.updated_node.left_child, LeafNode)
        self.assertIsInstance(proposal.updated_node.right_child, LeafNode)
        self.assertIsInstance(proposal.existing_node, LeafNode)


if __name__ == '__main__':
    unittest.main()
