import statsmodels.api as sm
from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def plot_qq(model: SklearnModel) -> None:
    residuals = model.residuals()
    fig = sm.qqplot(residuals, fit=True, line="45")
    plt.show()
