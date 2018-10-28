import statsmodels.api as sm
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def plot_qq(model: SklearnModel, ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    residuals = model.residuals()
    sm.qqplot(residuals, fit=True, line="45", ax=ax)
    plt.show()
