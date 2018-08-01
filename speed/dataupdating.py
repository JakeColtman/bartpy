from datetime import datetime as dt
import timeit
import cProfile
from copy import deepcopy

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.model import Model
from unittest import TestCase

from bartpy.tree import TreeMutation, TreeStructure, TreeNode, LeafNode, SplitNode, TreeMutation, PruneMutation, split_node
from bartpy.data import Data, Split, SplitCondition, LTESplitCondition

import pandas as pd

import pandas as pd
import numpy as np


def update_data():
    
    data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series([1, 2, 3]))
    a = split_node(LeafNode(data), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    new_data = deepcopy(data)
    new_data._y = pd.Series(np.random.uniform(0, 3, size=3))

    for _ in range(50 * 200):
        tree_structure.update_data(new_data)


def predict():

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)
    for ii in range(10):
        model.predict()


if __name__ == "__main__":
    print(timeit.timeit("update_data()", number=1, setup='from speed.dataupdating import update_data'))
    #print(timeit.timeit("predict()", number=1, setup='from speed.dataupdating import predict'))
