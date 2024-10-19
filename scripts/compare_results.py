import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def plot_committor_estimate(fig, ax, file_in, nb_runs=1, levels = None):
        data = np.loadtxt(file_in, delimiter=";")
        t_init = data[:, 0]
        x_init = data[:, 1]
        probabilities = data[:, 2]
        probabilities[probabilities == 0] = 1e-22   # Avoid log(0)
        errors = data[:, 3]
        resolution = [np.size(np.unique(t_init)), np.size(np.unique(x_init))]
        fig.suptitle(f'AMS: runs = {nb_runs}, resolution = {resolution}')
        if levels is None:
            levels = np.logspace(-6,0,num=25)
        t=ax.tricontourf(
            t_init, x_init, probabilities,
            alpha=0.7, cmap="viridis",
            levels=levels,
            norm=LogNorm() 
        )
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel(r"$x_{init}$")
        ax.grid(True)
        cbar = fig.colorbar(t)
        #cbar.set_label("Committor estimate")
        return fig, ax


def plot_PB(fig, ax, model):
    PB_traj = model.get_pullback(return_between_equil = True)
    ax.plot(
            PB_traj[:, 0],
            PB_traj[:, 1],
            label="PB_attractor",
            color="darkblue", linewidth=2
            )
    return fig, ax

#---------------------------------------------------------------

from DoubleWell_Model import DoubleWell_1D
model = DoubleWell_1D(mu = 0.03)
file_committor = "../temp/simulationAMS_runs1_grid1764.txt"
fig, ax = plt.subplots(dpi=250)
fig, ax = plot_PB(fig, ax, model)
fig, ax = plot_committor_estimate(fig, ax, file_committor, nb_runs=1)

file_out = "../temp/committor_vs_PB.png"
fig.savefig(file_out)
print('Saving to...', file_out)
