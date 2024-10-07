import numpy as np
from DoubleWell_Model import DoubleWell_1D


class PB_attractor:
    def __init__(self, model, mu, noise_factor=0):
        self.model = model
        self.mu = mu
        self.noise_factor = noise_factor

    def estimate_MC(self, N_traj=100, T_max=400, dt=0.01, t_0=-200):
        t_init = np.full(N_traj, t_0)
        x_init = np.linspace(-2, 2, N_traj)
        initial_state = np.stack([t_init, x_init], axis=1)
        traj = self.model.trajectory(
            N_traj, T_max, dt, self.mu, self.noise_factor, initial_state
        )
        traj = np.mean(traj, axis=0)
        self.PB_traj = traj
