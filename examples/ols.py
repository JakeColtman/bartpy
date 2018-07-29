from bartpy.data import Data
import pandas as pd
from bartpy.sigma import Sigma
from bartpy.model import Model
from bartpy.sampler import Sampler
from bartpy.proposer import Proposer


data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
sigma = Sigma(1., 2.)
model = Model(data, sigma, n_trees=10)

proposer = Proposer(0.5, 0.5, 0)
sampler = Sampler(model, proposer)


print(data.y)

print(model.predict())
preds = model.predict()
for ii in range(100):
    sampler.step()
    preds = model.predict()
    print(ii)
print(preds)