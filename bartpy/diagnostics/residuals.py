from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from bartpy.sklearnmodel import SklearnModel


def plot_qq(model: SklearnModel, ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    residuals = model.residuals(model.data.X)
    sm.qqplot(residuals, fit=True, line="45", ax=ax)
    ax.set_title("QQ plot")


def plot_homoskedasity_diagnostics(model: SklearnModel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.regplot(model.predict(model.data.X), model.residuals(model.data.X))
    ax.set_title("Fitted Values V Residuals")
    ax.set_xlabel("Fitted Value")
    ax.set_ylabel("Residual")
    return ax
