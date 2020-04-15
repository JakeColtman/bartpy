import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from bartpy.sklearnmodel import SklearnModel

def run(alpha, beta, n_trees, size=100):
    import warnings

    warnings.simplefilter("error", UserWarning)
    x = np.linspace(0, 5, size)
    y = np.random.normal(0, 1.0, size=size) + np.sin(x)
    X = pd.DataFrame(x)
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


if __name__ == "__main__":
    run(0.95, 2., 200, size=1000)
