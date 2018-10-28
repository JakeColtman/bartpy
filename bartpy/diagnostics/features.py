from collections import Counter
import itertools
from copy import deepcopy
from typing import List, Mapping, Union

from joblib import Parallel
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bartpy.model import Model
from bartpy.sklearnmodel import SklearnModel

ImportanceMap = Mapping[str, float]
ImportanceDistributionMap = Mapping[str, List[float]]


def feature_split_proportions_counter(model_samples: List[Model]) -> Mapping[int, float]:

    split_variables = []
    for sample in model_samples:
        for tree in sample.trees:
            for node in tree.nodes:
                splitting_var = node.split.splitting_variable
                split_variables.append(splitting_var)
    return {x[0]: x[1] / len(split_variables) for x in Counter(split_variables).items() if x[0] is not None}


def plot_feature_split_proportions(model_samples: List[Model]):
    proportions = feature_split_proportions_counter(model_samples)

    y_pos = np.arange(len(proportions))
    name, count = list(proportions.keys()), list(proportions.values())

    plt.barh(y_pos, count, align='center', alpha=0.5)
    plt.yticks(y_pos, name)
    plt.xlabel('Proportion of all splits')
    plt.ylabel('Feature')
    plt.title('Proportion of Splits Made on Each Variable')
    plt.show()


def null_feature_split_proportions_distribution(model: SklearnModel,
                                                X: Union[pd.DataFrame, np.ndarray],
                                                y: np.ndarray,
                                                n_permutations: int=10) -> Mapping[int, List[float]]:
    """
    Calculate a null distribution of proportion of splits on each variable in X

    Works by randomly permuting y to remove any true dependence of y on X and calculating feature importance

    Parameters
    ----------
    model: SklearnModel
        Model specification to work with
    X: np.ndarray
        Covariate matrix
    y: np.ndarray
        Target data
    n_permutations: int
        How many permutations to run
        The higher the number of permutations, the more accurate the null distribution, but the longer it will take to run
    Returns
    -------
    Mapping[int, List[float]]
        A list of inclusion proportions for each variable in X
    """

    inclusion_dict = {x: [] for x in range(X.shape[1])}

    delayed_chains = []
    for _ in range(n_permutations):
        permuted_model = deepcopy(model)
        y_perm = np.random.permutation(y)
        delayed_chains += permuted_model.delayed_chains(X, y_perm)

    n_jobs = model.n_jobs
    combined_samples = Parallel(n_jobs)(delayed_chains)
    combined_model_samples = [x[0] for x in combined_samples]
    flattened_model_samples = list(itertools.chain.from_iterable(combined_model_samples))
    by_run_model_samples = np.array_split(flattened_model_samples, n_permutations)

    for run_samples in by_run_model_samples:
        splits_run = feature_split_proportions_counter(run_samples)
        for key, value in splits_run.items():
            inclusion_dict[key].append(value)

    return inclusion_dict


def plot_null_feature_importance_distributions(null_distributions: Mapping[str, List[float]], ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df = pd.DataFrame(null_distributions)
    df = pd.DataFrame(df.unstack()).reset_index().drop("level_1", axis=1)
    df.columns = ["variable", "p"]
    sns.boxplot(x="variable", y="p", data=df, ax=ax)
    ax.set_title("Null Feature Importance Distribution")
    

def local_thresholds(null_distributions: ImportanceDistributionMap, percentile: float) -> Mapping[str, float]:
    return {feature: np.percentile(null_distributions[feature], percentile) for feature in null_distributions}


def global_thresholds(null_distributions: ImportanceDistributionMap, percentile: float) -> Mapping[str, float]:
    q_s = []
    df = pd.DataFrame(null_distributions)
    for row in df.iter_rows():
        q_s.append(np.max(row))
    threshold = np.percentile(q_s, percentile)
    return {feature: threshold for feature in null_distributions}


def kept_features(feature_proportions, thresholds):
    kept_features = []
    for feature in feature_proportions:
        if feature_proportions[feature] > thresholds[feature]:
            kept_features.append(feature)
    return kept_features


def is_kept(feature_proportions, thresholds):
    return [feature_proportions[feature] > thresholds[feature] for feature in feature_proportions]


def partition_into_passed_and_failed_features(feature_proportions, thresholds):
    kept = kept_features(feature_proportions, thresholds)
    passed_features = {x[0]: x[1] for x in feature_proportions.items() if x[0] in kept}
    failed_features = {x[0]: x[1] for x in feature_proportions.items() if x[0] not in kept}
    return passed_features, failed_features


def plot_feature_proportions_against_thresholds(feature_proportions, thresholds, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    passed_features, failed_features = partition_into_passed_and_failed_features(feature_proportions, thresholds)

    ax.bar(thresholds.keys(), [x * 100 for x in thresholds.values()], width=0.01, color="black", alpha=0.5)
    ax.scatter(passed_features.keys(), [x * 100 for x in passed_features.values()], c="g")
    ax.scatter(failed_features.keys(), [x * 100 for x in failed_features.values()], c="r")
    ax.set_title("Feature Importance Compared to Threshold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("% Splits")