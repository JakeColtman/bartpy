import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from bartpy.extensions.baseestimator import ResidualBART


def run(size=100,
        alpha=0.95,
        beta=2.0,
        n_trees=50):

    import warnings

    warnings.simplefilter("error", UserWarning)
    x = np.linspace(0, 5, size)
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=size) + np.sin(x)

    model = ResidualBART(
                n_samples=100,
                n_burn=50,
                n_trees=n_trees,
                alpha=alpha,
                beta=beta,
                n_jobs=1,
                n_chains=1)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42,
                                                        shuffle=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.show()

    rmse = np.sqrt(np.sum(np.square(y_test - y_pred)))
    print(rmse)


if __name__ == "__main__":
    run(50, 0.95, 2.0)
