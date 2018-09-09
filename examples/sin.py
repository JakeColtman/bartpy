import pandas as pd
import numpy as np

from bartpy.sklearnmodel import SklearnModel
from bartpy.plotting import plot_residuals, plot_modelled_against_actual


def run(alpha, beta, n_trees):
    x = np.sin(np.linspace(0, 5, 300))
    X = pd.DataFrame({"b": x})
    y = np.random.normal(0, 0.1, size=300) + x

    model = SklearnModel(n_samples=20, n_burn=20, n_trees=n_trees, alpha = alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict()
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", 2 * np.sin(x[ii]))

    #plot_residuals(model)
    #plot_modelled_against_actual(model)

    return model


if __name__ == "__main__":
    print("here")
    #run(0.95, 2., 20)