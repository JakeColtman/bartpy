from copy import deepcopy
from datetime import datetime
import timeit

from bartpy.data import Data
from bartpy.model import Model
from bartpy.sigma import Sigma
from bartpy.split import SplitCondition
from bartpy.tree import TreeStructure, LeafNode, TreeMutation, split_node

import pandas as pd
import numpy as np


def update_data():

    print(datetime.now())
    data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series([1, 2, 3]))
    a = split_node(LeafNode(data), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    new_data = deepcopy(data)
    new_data._y = pd.Series(np.random.uniform(0, 3, size=3))
    print(datetime.now())

    for _ in range(50 * 20):
        tree_structure.update_y(new_data)


def predict():


    print(datetime.now())
    data = Data(pd.DataFrame({"a": np.random.normal(0, 1, 10000), "b": np.random.normal(0, 1, 10000)}), pd.Series(np.random.normal(0, 1, 10000)))
    a = split_node(LeafNode(data), SplitCondition("a", 1))
    tree_structure = TreeStructure(a)

    c = split_node(a._right_child, SplitCondition("b", 2))
    tree_structure.mutate(TreeMutation("grow", a.right_child, c))

    new_data = deepcopy(data)
    new_data._y = pd.Series(np.random.normal(0, 1, 10000))
    print(datetime.now())

    for _ in range(50 * 20):
        tree_structure.update_y(new_data)
        tree_structure.predict()



if __name__ == "__main__":
    #print(timeit.timeit("update_data()", number=1, setup='from speed.dataupdating import update_data'))
    print(timeit.timeit("predict()", number=1, setup='from speed.dataupdating import predict'))
