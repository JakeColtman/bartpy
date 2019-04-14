import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel
from bartpy.diagnostics.trees import plot_tree_depth
from bartpy.diagnostics.features import plot_feature_split_proportions
from bartpy.diagnostics.residuals import plot_qq


def run(alpha, beta, n_trees):
    x = np.linspace(0, 5, 300)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=300) + np.sin(x)

    model = SklearnModel(n_samples=50, n_burn=50, n_trees=n_trees, alpha=alpha, beta=beta, n_jobs=1)
    model.fit(X, y)
    plt.plot(model.data.unnormalized_y)
    plt.plot(model.predict(X))
    plt.show()
    plot_tree_depth(model)
    plot_feature_split_proportions(model)
    plot_qq(model)
    #null_distr = null_feature_split_proportions_distribution(model, X, y)
    #print(null_distr)
    return model, x, y


if __name__ == "__main__":
    import cProfile
    from datetime import datetime as dt
    print(dt.now())
    run(0.95, 2., 50)
    #cProfile.run("run(0.95, 2., 200)")
    print(dt.now())
