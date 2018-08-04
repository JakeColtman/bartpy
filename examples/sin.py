import pandas as pd
import numpy as np

from bartpy.sklearnmodel import SklearnModel
from bartpy.plotting import plot_residuals, plot_modelled_against_actual

if __name__ == "__main__":

    x = np.random.normal(0, 5, size=300)
    x.sort()
    X = pd.DataFrame({"b": x})
    y = np.random.normal(0, 0.1, size=300) + 2 * np.sin(x)

    model = SklearnModel(n_samples=10, n_burn=10)
    model.fit(X, y)
    predictions = model.predict()
    for ii in range(len(predictions)):
        print(predictions[ii], " - ", 2 * np.sin(x[ii]))

    plot_residuals(model)
    plot_modelled_against_actual(model)