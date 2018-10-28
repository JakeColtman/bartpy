from matplotlib import pyplot as plt

from bartpy.diagnostics.residuals import plot_qq
from bartpy.diagnostics.trees import plot_tree_depth
from bartpy.sklearnmodel import SklearnModel


def plot_diagnostics(model: SklearnModel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    plot_qq(model, ax1)
    plot_tree_depth(model.model_samples, ax2)
    plt.show()