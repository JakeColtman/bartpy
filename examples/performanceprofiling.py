import pandas as pd
import numpy as np

from bartpy.sklearnmodel import SklearnModel


def ols_with_all_unique_columns(alpha, beta, n_trees, n_regressors, n_burn=50, n_samples=200, n_obsv=1000):
    b_true = np.random.uniform(-2, 2, size = n_regressors)
    x = np.random.normal(0, 1, size=n_obsv * n_regressors).reshape(n_obsv, n_regressors)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=n_obsv) + np.array(X.multiply(b_true, axis = 1).sum(axis=1))
    model = SklearnModel(n_samples=n_samples,
                         n_burn=n_burn,
                         n_trees=n_trees,
                         alpha=alpha,
                         beta=beta,
                         n_jobs=1,
                         n_chains=1,
                         initializer=None,
                         store_acceptance_trace=False,
                         store_in_sample_predictions=False)
    model.fit(X, y)
    return model, x, y


def ols_with_all_small_amount_of_duplication(alpha, beta, n_trees, n_regressors, n_burn=50, n_samples=200, n_obsv=1000):
    b_true = np.random.uniform(-2, 2, size = n_regressors)
    x = np.random.normal(0, 1, size=n_obsv * n_regressors).reshape(n_obsv, n_regressors)

    x[:, :5] = 4

    X = pd.DataFrame(x)

    y = np.random.normal(0, 0.1, size=n_obsv) + np.array(X.multiply(b_true, axis = 1).sum(axis=1))
    model = SklearnModel(n_samples=n_samples,
                         n_burn=n_burn,
                         n_trees=n_trees,
                         alpha=alpha,
                         beta=beta,
                         n_jobs=1,
                         n_chains=1,
                         initializer=None,
                         store_acceptance_trace=False,
                         store_in_sample_predictions=False)
    model.fit(X, y)
    return model, x, y


def ols_with_significant_categorical_variables(alpha, beta, n_trees, n_regressors, n_burn=50, n_samples=200, n_obsv=1000):
    b_true = np.random.uniform(-2, 2, size = n_regressors)
    x = np.random.normal(0, 1, size=n_obsv * n_regressors).reshape(n_obsv, n_regressors)

    for i in range(n_regressors //2):
        x[:, i] = np.random.binomial(3, 0.5, size=x.shape[0])

    X = pd.DataFrame(x)

    y = np.random.normal(0, 0.1, size=n_obsv) + np.array(X.multiply(b_true, axis = 1).sum(axis=1))
    model = SklearnModel(n_samples=n_samples,
                         n_burn=n_burn,
                         n_trees=n_trees,
                         alpha=alpha,
                         beta=beta,
                         n_jobs=1,
                         n_chains=1,
                         initializer=None,
                         store_acceptance_trace=False,
                         store_in_sample_predictions=False)
    model.fit(X, y)
    return model, x, y




if __name__ == "__main__":
    from timeit import default_timer as timer

    all_unique_small_n, all_unique_big_n, some_duplication_small_n, significant_categorical_variables = [None] * 4


    start = timer()
    all_unique_small_n = ols_with_all_unique_columns(0.95, 2., 200, 50, n_obsv=100)
    end = timer()
    all_unique_small_n = end - start

    start = timer()
    all_unique_big_n = ols_with_all_unique_columns(0.95, 2., 200, 50, n_obsv=100000)
    end = timer()
    all_unique_big_n = end - start
    #
    # start = timer()
    # some_duplication_small_n = ols_with_all_small_amount_of_duplication(0.95, 2., 200, 50, n_obsv=100)
    # end = timer()
    # some_duplication_small_n = end - start
    #
    # start = timer()
    # significant_categorical_variables = ols_with_significant_categorical_variables(0.95, 2., 200, 50, n_obsv=10000)
    # end = timer()
    # significant_categorical_variables = end - start

    print({
        "all_unique_small_n": all_unique_small_n,
        "all_unique_big_n": all_unique_big_n,
        "some_duplication_small_n": some_duplication_small_n,
        "significant_categorical_variables": significant_categorical_variables,
    })