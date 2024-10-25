import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import textwrap
from PB_attractor import PB_attractor
from DoubleWell_Model import DoubleWell_1D
    

class plot_DoubleWell:
    def __init__(self, t, mu):
        self.mu = mu
        self.t = t

    def V(self, x):
        return 0.25 * x**4 - 0.5 * x**2 - self.mu * x * self.t

    def plot(self):
        x = np.linspace(-2, 2, 100)
        fig, ax = plt.subplots()
        ax.plot(x, self.V(x), label="mu = " + str(self.mu) + ", t = " + str(self.t))
        ax.set_xlabel("x")
        ax.set_ylabel("V(x)")
        ax.legend()
        fig.savefig(f"../results/figures/DoubleWell_potential_t_{self.t:.0f}" + ".png")


class plot_trajectories:
    def __init__(self, traj, params: dict, filename: str):
        self.traj = traj
        self.params = params
        self.filename = filename

    def plot(self):
        fig, ax = plt.subplots(dpi=200)
        fig.suptitle(str(self.params))
        for i in range(self.traj.shape[0]):
            ax.plot(
                self.traj[i, :, 0], self.traj[i, :, 1], label="Trajectory " + str(i + 1)
            )
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position x")
        ax.grid()
        # ax.legend()
        fig.savefig(self.filename)


class plot_committor:
    def __init__(self, file_in: str):
        self.file_in = file_in

    def read_in_data(self):
        parameters = {}
        simulation_values = []
        with open(self.file_in, "r") as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if ":" in line:
                    key, value = line.split(":")
                    parameters[key.strip()] = float(value.strip())
                elif ";" in line:
                    simulation_values.append([float(i) for i in line.split(";")])
        float_array = np.array(simulation_values)
        self.parameters = parameters
        self.results = float_array
        self.t_init = self.results[:, 0]
        self.x_init = self.results[:, 1]
        self.probabilities = self.results[:, 2]
        self.probabilities[self.probabilities == 0] = 1e-22   # Avoid log(0)
        self.errors = self.results[:, 3]
        self.resolution = [np.size(np.unique(self.t_init)), np.size(np.unique(self.x_init))]

    def plot(self, fig, ax, add_title=True, levels=None):
        if add_title==True:
            wrapped_title = "\n".join(
                textwrap.wrap("TAMS: " + str(self.parameters), width=70)
            )
            fig.subplots_adjust(top=0.85)
            fig.text(0.5, 0.95, wrapped_title, ha="center", va="top", fontsize=10)
        if levels is None:
            self.levels = np.logspace(-4,0,num=15)
        self.levels = levels
        t=ax.tricontourf(
            self.t_init, self.x_init, self.probabilities,
            alpha=0.7, cmap="viridis",
            levels=self.levels,
            norm=LogNorm() 
        )
        '''ax.tricontour(
            self.t_init, self.x_init, self.probabilities,
            colors='black', linewidths=0.5,
            levels=self.levels,
            norm=LogNorm()
        )'''
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel(r"$x_{init}$")
        ax.grid()
        cbar = fig.colorbar(t)
        #cbar.set_label("Committor estimate")



class plot_PB:
    def __init__(self, mu: float):
        self.model = DoubleWell_1D()
        self.mu = mu

    def get_PB(self):
        pullback = PB_attractor(self.model, self.mu)
        pullback.estimate_MC()
        self.PB_traj = pullback.PB_traj
    
    def plot(self, fig, ax):
        ax.plot(
            self.PB_traj[:, 0],
            self.PB_traj[:, 1],
            label=f"PB_attractor with mu={self.mu}",
            color="darkblue", linewidth=2
            )
    

class plot_uncertainties():
    def __init__(self, file_in: str):
        self.file_in = file_in

    def read_in_data(self):
        parameters = {}
        simulation_values = []
        with open(self.file_in, "r") as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if ":" in line:
                    key, value = line.split(":")
                    parameters[key.strip()] = float(value.strip())
                elif ";" in line:
                    simulation_values.append([float(i) for i in line.split(";")])
        float_array = np.array(simulation_values)
        self.parameters = parameters
        self.results = float_array
        self.probabilities = self.results[:, 2]
        self.probabilities[self.probabilities == 0] = 1e-22   # Avoid log(0)
        self.errors = self.results[:, 3]
        

    def plot(self, fig, ax):
        wrapped_title = "\n".join(
            textwrap.wrap("TAMS: " + str(self.parameters), width=70)
        )
        fig.subplots_adjust(top=0.85)
        fig.text(0.5, 0.95, wrapped_title, ha="center", va="top", fontsize=10)
        ax.errorbar(
            self.t_init, self.probabilities, yerr=self.errors, fmt=".", label="Uncertainty",
            capsize=2, color='darkred'
        )
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel("Committor estimate")
        ax.set_yscale("log")
        ax.grid()


if __name__ == "__main__":

    file_in_committor = "../temp/simulationTAMS.txt"
    file_out = "../temp/comparison.png"

    tams_results = plot_committor(file_in_committor)
    tams_results.read_in_data()
    mu = tams_results.parameters["mu"]
    pullback = plot_PB(mu)
    pullback.get_PB()

    fig, ax = plt.subplots(dpi=250)
    levels=np.logspace(-6,0,num=25)
    tams_results.plot(fig, ax, levels)
    pullback.plot(fig, ax)
    ax.legend()
    ax.set_xlim(np.min(tams_results.t_init), np.max(tams_results.t_init))
    ax.set_ylim(np.min(tams_results.x_init), np.max(tams_results.x_init))
    fig.savefig(file_out)
    plt.close()
    print("Figure written to: ", file_out)
    print(f'Resolution in [t,x]: {tams_results.resolution}')

    '''
    fig, ax = plt.subplots(dpi=250)
    uncertainties = plot_uncertainties(file_in_committor)
    uncertainties.read_in_data()
    uncertainties.plot(fig, ax)
    ax.legend()
    fig.savefig("../results/figures/uncertainties.png")
    plt.show()
    '''
