import numpy as np
import matplotlib.pyplot as plt
import textwrap


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


class plot_probabilities:
    def __init__(self, file_in: str, filepath_out: str):
        self.file_in = file_in
        self.file_out = filepath_out

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
        self.errors = self.results[:, 3]

    def plot(self):
        fig, ax = plt.subplots(dpi=200)
        wrapped_title = "\n".join(
            textwrap.wrap("TAMS: " + str(self.parameters), width=70)
        )
        fig.subplots_adjust(top=0.85)
        fig.text(0.5, 0.95, wrapped_title, ha="center", va="top", fontsize=10)
        t = ax.tricontourf(
            self.t_init, self.x_init, self.probabilities, alpha=0.7, cmap="viridis"
        )
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel(r"$x_{init}$")
        ax.grid()
        cbar = fig.colorbar(t)
        cbar.set_label("Probability")
        self.ax = ax
        self.fig = fig

    def save(self, file_out):
        self.fig.savefig(file_out)


if __name__ == "__main__":

    from PB_attractor import PB_attractor
    from DoubleWell_Model import DoubleWell_1D

    file_in = "../results/outputs/keep/simulationTAMS_0410.txt"
    filepath_out = "../results/figures/"
    tams_results = plot_probabilities(file_in, filepath_out)
    tams_results.read_in_data()
    mu = tams_results.parameters["mu"]
    model = DoubleWell_1D()
    PB_attractor_0 = PB_attractor(
        model, mu=tams_results.parameters["mu"], noise_factor=0
    )
    PB_attractor_0.estimate_MC()
    PB_attractor_1 = PB_attractor(model, mu=0.01, noise_factor=0.1)
    PB_attractor_1.estimate_MC()
    tams_results.plot()
    tams_results.save(filepath_out + "commitor_estimate.png")
    tams_results.ax.plot(
        PB_attractor_0.PB_traj[:, 0],
        PB_attractor_0.PB_traj[:, 1],
        label=f"PB_attractor with mu={PB_attractor_0.mu}",
        color="darkred",
    )
    # tams_results.ax.plot(PB_attractor_1.PB_traj[:,0], PB_attractor_1.PB_traj[:,1], label=f'PB_attractor with mu={PB_attractor_1.mu}',color='darkblue')
    tams_results.ax.legend()
    tams_results.ax.set_xlim(-50, 160)
    tams_results.save(filepath_out + "commitor_vs_PB.png")
