import pandas as pd
import numpy as np

from bartpy.sklearnmodel import SklearnModel

if __name__ == "__main__":

    x = np.random.normal(0, 5, size=3000)
    x.sort()
    X = pd.DataFrame({"b": x})
    y = np.random.normal(0, 0.1, size=3000) + 2 * x

    model = SklearnModel(n_samples=10, n_burn=1)
    model.fit(X, y)
    predictions = model.predict()
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", 2 * x[ii])
