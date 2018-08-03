from copy import deepcopy
from datetime import datetime
import timeit

from bartpy.data import Data
from bartpy.model import Model
from bartpy.sigma import Sigma
from bartpy.split import SplitCondition, Split
from bartpy.tree import TreeStructure, LeafNode, TreeMutation, split_node

import pandas as pd
import numpy as np


def update_data():
    print(datetime.now())
    data = Data(pd.DataFrame({"a": np.random.normal(0, 1, 10000), "b": np.random.normal(0, 1, 10000)}), pd.Series(np.random.normal(0, 1, 10000)))
    split = Split(data, [])
    a = split_node(LeafNode(split), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    new_y = pd.Series(np.random.normal(0, 1, 10000))
    print(datetime.now())

    for _ in range(50 * 2000):
        tree_structure.update_y(new_y)


def predict():


    print(datetime.now())
    data = Data(pd.DataFrame({"a": np.random.normal(0, 1, 10000), "b": np.random.normal(0, 1, 10000)}), pd.Series(np.random.normal(0, 1, 10000)))
    split = Split(data, [])
    a = split_node(LeafNode(split), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    new_y = pd.Series(np.random.normal(0, 1, 10000))
    print(datetime.now())

    for _ in range(50 * 2000):
        tree_structure.update_y(new_y)
        tree_structure.predict()




def updated_trees():

    print(datetime.now())
    data = Data(pd.DataFrame({"a": np.random.normal(0, 1, 10000), "b": np.random.normal(0, 1, 10000)}), pd.Series(np.random.normal(0, 1, 10000)))
    split = Split(data, [])
    a = split_node(LeafNode(split), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))
    sigma = Sigma(100., 0.001)

    model = Model(data=data, trees=[deepcopy(tree_structure) for x in range(100)], sigma=sigma)
    print(datetime.now())

    for _ in range(500):
        for _ in model.refreshed_trees():
            pass



if __name__ == "__main__":
    print(timeit.timeit("updated_trees()", number=1, setup='from speed.dataupdating import updated_trees'))
    #print(timeit.timeit("predict()", number=1, setup='from speed.dataupdating import predict'))
