from datetime import datetime as dt
import timeit
import cProfile
from copy import deepcopy

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.model import Model

import pandas as pd
import numpy as np


def update_data():

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)
    for ii in range(10):

        for tree in model.trees:
            new_data = deepcopy(data)
            new_data._y = pd.Series(np.random.uniform(0, 3, size=3))
            tree.update_data(new_data)


def predict():

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)
    for ii in range(10):
        model.predict()







if __name__ == "__main__":
    print(timeit.timeit("update_data()", number=1, setup='from speed.dataupdating import update_data'))
    print(timeit.timeit("predict()", number=1, setup='from speed.dataupdating import predict'))
