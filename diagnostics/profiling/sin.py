import pandas as pd
import numpy as np
from bartpy.sklearnmodel import SklearnModel

def run(alpha, beta, n_trees, size=100):
    x = np.linspace(0, 5, size)
    y = np.random.normal(0, 1.0, size=size) + np.sin(x)
    X = pd.DataFrame(x)

    model = SklearnModel(
                n_samples=50,
                n_burn=50,
                n_trees=n_trees,
                alpha=alpha,
                beta=beta,
                n_jobs=1,
                n_chains=1)
    model.fit(X, y)
    model.predict(X)

if __name__ == "__main__":
    run(0.95, 2., 200, size=1000)
