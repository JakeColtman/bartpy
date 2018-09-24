import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel
from bartpy.plotting import plot_residuals, plot_modelled_against_actual


def run(alpha, beta, n_trees, n_regressors):
    b_true = np.random.uniform(-2, 2, size = n_regressors)
    x = np.random.normal(0, 1, size=10000 * n_regressors).reshape(10000, n_regressors)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=10000) + np.array(X.multiply(b_true, axis = 1).sum(axis=1))
    model = SklearnModel(n_samples=200, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict()
    plt.scatter(y, predictions)
    plt.show()
    return model, x, y


if __name__ == "__main__":
    import cProfile
    from datetime import datetime as dt
    print(dt.now())
    #model, x, y = run(0.95, 2., 50, 5)
    cProfile.run("run(0.95, 2., 30, 5)")

    print(dt.now())
