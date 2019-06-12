import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from bartpy.diagnostics.features import feature_split_proportions, null_feature_split_proportions_distribution, local_thresholds, plot_null_feature_importance_distributions, \
    plot_feature_proportions_against_thresholds
from bartpy.sklearnmodel import SklearnModel


def run(n: int=1000, k_true: int=5, k_null: int=2):
    b_true = np.random.uniform(2, 0.1, size=k_true)
    b_true = np.array(list(b_true) + [0.0] * k_null)
    print(len(b_true))
    x = np.random.normal(0, 1, size=n * (k_true + k_null)).reshape(n, (k_true + k_null))

    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=n) + np.array(X.multiply(b_true, axis=1).sum(axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42,
                                                        shuffle=True)

    model = SklearnModel(n_samples=50,
                         n_burn=50,
                         n_trees=15,
                         alpha=0.8,
                         beta=2.,
                         n_chains=1,
                         n_jobs=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.show()

    rmse = np.sqrt(np.sum(np.square(y_test - y_pred)))
    feature_proportions = feature_split_proportions(model, list(range(X.shape[1])))
    print(feature_proportions)

    null_distribution = null_feature_split_proportions_distribution(model, X_train, y_train, 10)
    print(null_distribution)
    thresholds = local_thresholds(null_distribution, 0.75)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plot_feature_proportions_against_thresholds(feature_proportions, thresholds, ax1)
    plot_null_feature_importance_distributions(null_distribution, ax2)
    plt.show()


if __name__ == "__main__":
    run(1000, 5, 2)
