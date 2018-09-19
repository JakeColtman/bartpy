import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

from bartpy.extensions.ols import OLS
from bartpy.plotting import plot_residuals, plot_modelled_against_actual


def run(alpha, beta, n_trees):
    x = np.random.normal(0, 1, size=3000)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=3000) + 2 * x + np.sin(x)
    plt.scatter(x, y)
    plt.show()
    model = OLS(stat_model=sm.OLS, n_samples=200, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict()
    plot_residuals(model)
    plot_modelled_against_actual(model)
    plt.scatter(x, y)
    plt.scatter(x, predictions)
    return model, x, y


if __name__ == "__main__":
    print("here")
    from datetime import datetime as dt
    print(dt.now())
    model, x, y = run(0.95, 2., 50)
    print(model.stat_model_fit.summary())
    print(dt.now())
