import unittest

from bartpy.data import Data
from bartpy.samplers.treemutation.uniform.proposer import UniformMutationProposer
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
        self.tree = Tree([self.a, self.b, self.c, self.d, self.e])
        self.proposer = UniformMutationProposer()

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.propose(self.tree)
        self.assertIn(proposal.existing_node, self.tree.nodes)
        self.assertNotIn(proposal.updated_node, self.tree.nodes)

    def test_types(self):
        proposal = self.proposer.propose(self.tree)
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
        self.tree = Tree([self.a, self.b, self.c, self.d, self.e])
        self.proposer = UniformMutationProposer()

    def test_proposal_isnt_mutating(self):
        proposal = self.proposer.propose(self.tree)
        self.assertIn(proposal.existing_node, self.tree.nodes)
        self.assertNotIn(proposal.updated_node, self.tree.nodes)

    def test_types(self):
        proposal = self.proposer.propose(self.tree)
        self.assertIsInstance(proposal.updated_node, DecisionNode)
        self.assertIsInstance(proposal.updated_node.left_child, LeafNode)
        self.assertIsInstance(proposal.updated_node.right_child, LeafNode)
        self.assertIsInstance(proposal.existing_node, LeafNode)


if __name__ == '__main__':
    unittest.main()
