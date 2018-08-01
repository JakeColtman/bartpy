from bartpy.data import Data
import pandas as pd
from bartpy.sigma import Sigma
from bartpy.model import Model
from bartpy.sampler import Sampler
from bartpy.proposer import Proposer

if __name__ == "__main__":

    data = Data(pd.DataFrame({"b": [1, 2, 3, 4, 5, 6, 7] * 7}), pd.Series([1, 2, 3, 4, 5, 6, 7] * 7), normalize=True)
    sigma = Sigma(100., 0.001)
    model = Model(data, sigma, n_trees=20, k=2)

    proposer = Proposer(0.2, 0.2, 0.6)
    sampler = Sampler(model, proposer)

    print(data.y)

    print(model.predict())

    s = sampler.samples(200, 200)
    predictions = s.mean(axis=0)
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", data.y[ii])
