import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from bartpy.initializers.initializer import Initializer
from bartpy.sklearnmodel import SklearnModel
# from bartpy.diagnostics.trees import plot_tree_depth
# from bartpy.diagnostics.features import plot_feature_split_proportions
# from bartpy.diagnostics.residuals import plot_qq
#
# from bartpy.samplers.oblivioustrees.treemutation import get_tree_sampler
# import statsmodels.api as sm
# from bartpy.extensions.baseestimator import ResidualBART
# from bartpy.extensions.ols import OLS
#

def run(alpha, beta, n_trees, size=100):
    import warnings

    warnings.simplefilter("error", UserWarning)
    x = np.linspace(0, 5, size)
    y = np.random.normal(0, 1.0, size=size) + np.sin(x)
    X = pd.DataFrame(x)
    # X = torch.from_numpy(x.reshape(-1, 1))
    # y = torch.from_numpy(y)
    from bartpy.samplers.unconstrainedtree.treemutation import get_tree_sampler

    model = SklearnModel(
                n_samples=50,
                n_burn=50,
                n_trees=n_trees,
                alpha=alpha,
                beta=beta,
                n_jobs=1,
                n_chains=1, tree_sampler=get_tree_sampler(0.5, 0.5))
    model.fit(X, y)
    plt.plot(y)
    plt.plot(model.predict(X))
    plt.show()
    # # plot_tree_depth(model)
    # plot_feature_split_proportions(model)
    # plot_qq(model)
    # null_distr = null_feature_split_proportions_distribution(model, X, y)
    # print(null_distr)
    return model, x, y


if __name__ == "__main__":
    import cProfile
    from datetime import datetime as dt
    print(dt.now())

    run(0.95, 2., 200, size=50000)
    #cProfile.run("run(0.95, 2., 200, size=500)", "restatsto")
    print(dt.now())
