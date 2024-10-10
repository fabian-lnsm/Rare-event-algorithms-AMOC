import numpy as np


class DoubleWell_1D:
    """
    Class implementing the Double-Well model in 1D.
    Note: We simulate trajectories from left to right equilibrium point = from on to off
    """

    def __init__(self, on_state=-1.0, seed=None):

        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.on = on_state
        self.off = -on_state

    def is_on(self, traj):
        """
        Checks for on-states in a set of trajectories.
        On-state = left equibrium point of the bistable system

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 2)

        Returns
        -------
        An array of shape (Number of trajectories, timesteps)
        Every on-state is replaced with 1, all other states are replaced with 0

        """
        distance_to_on = traj[..., 1] - self.on
        return distance_to_on <= 0

    def is_off(self, traj):
        """
        Checks for off-states in a set of trajectories.
        Off-state = right equilibrium point of the bistable system

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 2)

        Returns
        -------
        An array of shape (Number of trajectories, timesteps)
        Every off-state is replaced with 1, all other states are replaced with 0

        """
        distance_to_off = traj[..., 1] - self.off
        return distance_to_off >= 0

    def number_of_transitioned_traj(self, traj):
        """
        Returns the number of transitioned trajectories.

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 2)

        Returns
        -------
        int : The number of trajectories that transitioned from on to off.
        """
        sum = np.sum(self.is_off(traj), axis=1)
        transitions = np.sum(sum > 0)
        return transitions

    def potential(self, t, x, mu):
        """
        The potential of the system at a given state.
        """
        return x**4 / 4 - x**2 / 2 - mu * x * t

    def force(self, t, x, mu):
        """
        The force of the system at a given state.
        """
        return -(x**3) + x + mu * t

    def euler_maruyama(
        self, t: np.ndarray, x: np.ndarray, dt: float, mu: float, noise_term: np.ndarray
    ):
        """
        Standard Euler-Maruyama integration scheme for the Double-Well model.

        Parameters
        ----------
        t : np.ndarray of shape (N_traj,)
            The current time of all trajectories.
        x : np.ndarray of shape (N_traj,)
            The current position of all trajectories.
        dt : float
            The time step of the integration.
        mu : float
            The coupling parameter of the system.
        noise_term : np.ndarray of shape (N_traj,)
            The noise term to add to the system.

        Returns
        -------
        t_new : np.ndarray of shape (N_traj,)
        x_new : np.ndarray of shape (N_traj,)
        """
        t_new = t + dt
        drift = self.force(t, x, mu) * dt
        x_new = x + drift + noise_term
        return t_new, x_new

    def trajectory(
        self,
        N_traj: int,
        T_max: int,
        dt: float,
        mu: float,
        noise_factor: float,
        init_state: np.ndarray,
        return_all: False
    ):
        """
        Compute trajectories of the system of fixed length using standard Euler integration.

        Parameters
        ----------
        N_traj : int
            The number of trajectories to compute.
        T_max : int
            The duration these traj should last.
        dt : float
            The time step of the integration
        mu : float
            The coupling parameter of the system
        noise_factor : float
            The amount of noise to add to the system
        init_state : state, thus of shape (N_traj, 2)
            The initial conditions for every trajectory.

        Returns
        -------
        simulated_trajectories : traj of shape (N_traj, T_max, 2)
            The computed trajectories downsampled to model time units.

        """

        n_steps = int(T_max / dt)
        trajectories = np.zeros((N_traj, n_steps, 2))
        t = init_state[:, 0]
        x = init_state[:, 1]
        trajectories[:, 0, 0] = t
        trajectories[:, 0, 1] = x
        noise = noise_factor * self.rng.normal(
            loc=0.0, scale=np.sqrt(dt), size=(N_traj, n_steps)
        )
        for i in range(1, n_steps):
            t, x = self.euler_maruyama(t, x, dt, mu, noise[:, i])
            trajectories[:, i, 0] = t
            trajectories[:, i, 1] = x

        if return_all==False:
            # Downsample the trajectory to return model time units
            i = int(1 / dt)
            trajectories = trajectories[:, ::i, :]
        return trajectories
    
    def get_pullback(self, mu: float, return_between_equil: bool = False, N_traj=100, T_max=400, dt=0.01, t_0=-200, noise_factor=0):
        '''
        return_between_equil:
        If true, returns the pullback trajectory only between t=0 and x>=1. Default: False
        '''
        t_init = np.full(N_traj, t_0)
        x_init = np.linspace(-2, 2, N_traj)
        initial_state = np.stack([t_init, x_init], axis=1)
        traj = self.trajectory(
            N_traj, T_max, dt, mu, noise_factor, initial_state, return_all=True
        )
        traj = np.mean(traj, axis=0)
        if return_between_equil==True:
            mask = (traj[:,0] >= 0) & (traj[:,1] <= 1.0)
            traj = traj[mask]
        self.PB_traj = traj
        return traj



if __name__ == "__main__":

    import time

    # set up the simulation
    model = DoubleWell_1D()
    N_traj = 10000  # number of trajectories per run
    tmax = 100  # maximum length of trajectories in model time units
    dt = 0.01  # time step of the simulation in model time units
    mu = 0.005  # coupling parameter in the model
    noise_factor = 0.1  # noise parameter in the model
    file_string = "../results/outputs/simulationMC.txt"
    params = {
        "N_traj": N_traj,
        "T_max": tmax,
        "dt": dt,
        "mu": mu,
        "noise_factor": noise_factor,
    }
    with open(file_string, "a") as f:
        f.write(" \n Simulation parameters: \n")
        f.write(str(params) + "\n" + "\n")
        f.write("Results: \n \n")
        f.write("IC Probability \n")

    # set initial conditions
    initial_condition = [0.0, -1.0]  # initial condition for [time,x]
    initial_condition = np.tile(
        initial_condition, (N_traj, 1)
    )  # turn into array of shape (N_traj,2)

    # run the simulation
    time_start = time.time()
    traj = model.trajectory(N_traj, tmax, dt, mu, noise_factor, initial_condition)
    probabilities = model.number_of_transitioned_traj(traj) / N_traj
    with open(file_string, "a") as f:
        f.write(f"{initial_condition[0,:]} {probabilities:.5f} \n")
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
    print("Probability of transition: ", probabilities)

    # plot the resulting trajectories
    from plot_functions import plot_trajectories

    params = {
        "N_traj": N_traj,
        "T_max": tmax,
        "dt": dt,
        "mu": mu,
        "noise_factor": noise_factor,
    }
    filename = "../results/figures/trajectoriesMC.png"
    plot_trajectories(traj, params, filename).plot()
