from bartpy.sklearnmodel import SklearnModel

from matplotlib import pyplot as plt


def plot_residuals(model: SklearnModel):
    plt.plot(model.data.unnormalized_y - model.predict())
    plt.show()


def plot_modelled_against_actual(model: SklearnModel):
    plt.plot(model.data.unnormalized_y)
    plt.plot(model.predict())
    plt.show()
