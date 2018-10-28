from matplotlib import pyplot as plt

from bartpy.diagnostics.residuals import plot_qq
from bartpy.diagnostics.sigma import plot_sigma_convergence
from bartpy.diagnostics.trees import plot_tree_depth
from bartpy.sklearnmodel import SklearnModel


def plot_diagnostics(model: SklearnModel):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    plot_qq(model, ax1)
    plot_tree_depth(model.model_samples, ax2)
    plot_sigma_convergence(model, ax3)
    plt.show()