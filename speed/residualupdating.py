from datetime import datetime as dt
import timeit
import cProfile

from bartpy.data import Data
from bartpy.sigma import Sigma
from bartpy.model import Model

import pandas as pd


def loop_refreshed_trees():

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)

    for ii in range(10):
        print(ii)
        for _ in model.refreshed_trees():
            pass


def loop_trees():

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)

    for ii in range(10):
        print(ii)
        for _ in model.trees:
            pass


if __name__ == "__main__":
    # print(cProfile.runctx("loop_refreshed_trees()", None, locals(), filename="refreshed_trees"))
    # import pstats
    #
    # p = pstats.Stats('refreshed_trees')
    # p.strip_dirs().sort_stats("tottime").print_stats()
    print(timeit.timeit("loop_trees()", number=1, setup='from speed.residualupdating import loop_trees'))
    print(timeit.timeit("loop_refreshed_trees()", number=1, setup='from speed.residualupdating import loop_refreshed_trees'))
