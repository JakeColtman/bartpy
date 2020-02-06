import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from bartpy.features.featureselection import SelectNullDistributionThreshold
from bartpy.sklearnmodel import SklearnModel


def run(n: int=10000, k_true: int=3, k_null: int=2):
    b_true = np.random.uniform(2, 0.1, size=k_true)
    b_true = np.array(list(b_true) + [0.0] * k_null)
    print(b_true)
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
                         n_trees=20,
                         store_in_sample_predictions=False,
                         n_jobs=3,
                         n_chains=1)

    pipeline = make_pipeline(SelectNullDistributionThreshold(model, n_permutations=20), model)
    pipeline_model = pipeline.fit(X_train, y_train)
    print("Thresholds", pipeline_model.named_steps["selectnulldistributionthreshold"].thresholds)
    print("Feature Proportions", pipeline_model.named_steps["selectnulldistributionthreshold"].feature_proportions)
    print("Is Kept", pipeline_model.named_steps["selectnulldistributionthreshold"]._get_support_mask())
    pipeline_model.named_steps["selectnulldistributionthreshold"].plot()


if __name__ == "__main__":
    run(1000, 5, 2)
