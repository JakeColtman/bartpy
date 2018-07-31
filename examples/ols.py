from bartpy.data import Data
import pandas as pd
from bartpy.sigma import Sigma
from bartpy.model import Model
from bartpy.sampler import Sampler
from bartpy.proposer import Proposer

if __name__ == "__main__":

    data = Data(pd.DataFrame({"b": [1, 2, 3, 4, 5, 6, 7, 8, 9]}), pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma, n_trees=10)

    proposer = Proposer(0.2, 0.2, 0.6)
    sampler = Sampler(model, proposer)

    print(data.y)

    print(model.predict())

    s = sampler.samples(100, 100)
    print(s)
    print(s.mean(axis=0))
