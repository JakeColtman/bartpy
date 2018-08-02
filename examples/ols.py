from bartpy.data import Data
import pandas as pd
from bartpy.sigma import Sigma
from bartpy.model import Model
from bartpy.sampler import Sampler
from bartpy.proposer import Proposer

if __name__ == "__main__":

    import numpy as np

    x = np.random.normal(0, 5, size=50)
    x.sort()
    y = np.random.normal(0, 0.1, size=50) + 2 * x
    data = Data(pd.DataFrame({"b": x}), pd.Series(y), normalize=True)
    sigma = Sigma(100., 0.001)
    model = Model(data, sigma, n_trees=50, k=2)

    proposer = Proposer(0.2, 0.2, 0.6)
    sampler = Sampler(model, proposer)

    print(data.y)

    print(model.predict())

    s = sampler.samples(50, 50)
    predictions = s.mean(axis=0)
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", data.y[ii])
