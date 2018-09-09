import pandas as pd
import numpy as np

from bartpy.sklearnmodel import SklearnModel
from bartpy.plotting import plot_residuals, plot_modelled_against_actual


def run(alpha, beta, n_trees):
    x = np.sin(np.linspace(0, 5, 100000))
    X = pd.DataFrame({"b": x})
    y = np.random.normal(0, 0.1, size=100000) + x

    model = SklearnModel(n_samples=50, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict()
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", 2 * np.sin(x[ii]))

    plot_residuals(model)
    plot_modelled_against_actual(model)

    return model, x, y


if __name__ == "__main__":
    print("here")
    import cProfile
    cProfile.run("run(0.95, 2., 50)")