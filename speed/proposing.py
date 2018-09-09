from copy import deepcopy
from datetime import datetime
import timeit

from bartpy.data import Data
from bartpy.model import Model
from bartpy.sigma import Sigma
from bartpy.split import SplitCondition, Split, sample_split_condition
from bartpy.tree import Tree, LeafNode, TreeMutation, split_node

import pandas as pd
import numpy as np


def propose_grow():

    print(datetime.now())
    data = Data(pd.DataFrame({"a": np.random.normal(0, 1, 10000), "b": np.random.normal(0, 1, 10000)}), pd.Series(np.random.normal(0, 1, 10000)))
    split = Split(data, [])
    a = split_node(LeafNode(split), SplitCondition("a", 1))
    tree_structure = Tree(a)

    c = split_node(a._right_child, SplitCondition("b", 0.5))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    print(datetime.now())

    from bartpy.proposer import TreeMutationProposer, ChangeTreeMutationProposer, GrowTreeMutationProposer, PruneTreeMutationProposer

    proposer = ChangeTreeMutationProposer(tree_structure)


    for _ in range(50 * 20):
        #tree_structure.random_splittable_leaf_node()

        #condition = sample_split_condition(a._left_child, None)
        #split_node(a, condition)
        #print(tree_structure.leaf_parents())
        #tree_structure.random_leaf_parent()
        proposer.proposal()


if __name__ == "__main__":
    #propose_grow()
    print(timeit.timeit("propose_grow()", number=1, setup='from speed.proposing import propose_grow'))
