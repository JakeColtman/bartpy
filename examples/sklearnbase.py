import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from bartpy.extensions.baseestimator import ResidualBART


def run(alpha, beta, n_trees):
    x = np.random.normal(0, 1, size=3000)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=3000) + 2 * x + np.sin(x)
    plt.scatter(x, y)
    plt.show()
    model = ResidualBART(n_samples=200, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta)
    model.fit(X, y)
    predictions = model.predict()
    plt.scatter(x, y)
    plt.scatter(x, predictions)
    plt.show()
    return model, x, y


if __name__ == "__main__":
    print("here")
    from datetime import datetime as dt
    print(dt.now())
    model, x, y = run(0.95, 2., 100)
    print(dt.now())
