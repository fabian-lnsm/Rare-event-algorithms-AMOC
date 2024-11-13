import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time as time


class DoubleWell_1D:
    """
    Class implementing the Double-Well model in 1D.
    Note: We simulate trajectories from left to right equilibrium point = from on to off
    """

    def __init__(self, mu : float, noise_factor : float = 0.1, dt : float = 0.01, seed=None):

        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.mu = mu
        self.dt = dt
        self.noise_factor = noise_factor


    def is_on(self, traj):
        """
        Checks for on-states in a set of trajectories.
        On-state = left equibrium point of the bistable system

        Parameters
        ----------
        traj : np.array of shape (Number of trajectories, timesteps, 2)

        Returns
        -------
        A np.array of shape (Number of trajectories, timesteps)
        Every on-state is replaced with 1, all other states are replaced with 0

        """

        time_current = np.round(traj[..., 0], 2)
        #time_current = np.round(traj[0, 0], 2)
        root_on = np.vectorize(self.on_dict.get)(time_current)
        on = traj[..., 1] <= root_on #left of the first root (on-state)
        return on

    def is_off(self, traj):
        """
        Checks for off-states in a set of trajectories.
        Off-state = right equilibrium point of the bistable system

        Parameters
        ----------
        traj : np.array of shape (Number of trajectories, timesteps, 2)

        Returns
        -------
        A np.array of shape (Number of trajectories, timesteps)
        Every off-state is replaced with 1, all other states are replaced with 0

        """ 
        time_current = np.round(traj[..., 0], 2)
        #time_current = np.round(traj[0, 0], 2)
        root_off = np.vectorize(self.off_dict.get)(time_current)
        off = root_off <= traj[..., 1]
        return off
        

    
    def set_roots(self, all_t):
        """
        Set the roots of the system for a given time interval.

        Parameters
        ----------
        all_t : np.array of shape (Number of timesteps,)
            The time points for which the roots should be computed.
        
        """
        roots = np.real(np.array([np.roots([-1, 0, 1, self.mu*t]) for t in all_t]))
        self.on_dict = dict(zip(all_t.T, roots[:, 1])) #left equilibrium point
        self.off_dict = dict(zip(all_t.T, roots[:, 0])) #right equilibrium point
    

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

    def potential(self, t, x):
        """
        The potential of the system at a given state.
        """
        return x**4 / 4 - x**2 / 2 - self.mu * x * t
    
    def plot_potential(self, t, ax):
        '''
        Plot the potential of the system at a given time.

        '''
        x = np.linspace(-2, 2, 1000)
        y = self.potential(t, x)
        ax.plot(x, y, label=f't={t}')
        ax.set_xlabel('x')
        ax.set_ylabel('V(x,t)')
        return ax

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

    def trajectory_TAMS(
        self,
        N_traj: int,
        T_max: int,
        init_state: np.ndarray,
        return_all = False,
        noise_factor = None,
    ):
        """
        Compute trajectories of the system of fixed length (TAMS) using Euler-Maruyama method.

        Parameters
        ----------
        N_traj : int
            The number of trajectories to compute.
        T_max : int
            The duration these traj should last.
        init_state : state, thus of shape (N_traj, 2)
            The initial conditions for every trajectory.
        return_all : bool
            Whether or not to return every timestep or only full steps. Default: False
        noise_factor : float
            If not provided, the noise factor of the model is used.

        Returns
        -------
        simulated_trajectories : traj of shape (N_traj, T_max, 2)
            The computed trajectories downsampled to model time units.
            
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        n_steps = int(T_max / self.dt)
        trajectories = np.zeros((N_traj, n_steps, 2))
        t = init_state[:, 0]
        x = init_state[:, 1]
        trajectories[:, 0, 0] = t
        trajectories[:, 0, 1] = x
        noise = noise_factor * self.rng.normal(
            loc=0.0, scale=np.sqrt(self.dt), size=(N_traj, n_steps)
        )
        for i in range(1, n_steps):
            t, x = self.euler_maruyama(t, x, self.dt, self.mu, noise[:, i])
            trajectories[:, i, 0] = t
            trajectories[:, i, 1] = x

        if return_all==False:
            # Downsample the trajectory to return model time units
            i = int(1 / self.dt)
            trajectories = trajectories[:, ::i, :]
        return trajectories
    
    def trajectory_AMS(
            self,
            N_traj: int,
            init_state: np.array,
            downsample: bool = False,
    ):
        """
        Compute trajectories of the system of variable length (AMS) using Euler-Maruyama method.

        Parameters
        ----------
        N_traj : int
            The number of trajectories to compute.
        init_state : state, thus of shape (N_traj, 2)
            The initial conditions for every trajectory.
        downsample: bool
            Whether to return every timestep or only full steps.
            true: return only full steps, false: return every timestep
            Default: True

        Returns
        -------
        simulated_trajectories : traj of shape (N_traj, time: not fixed, 2)
            The computed trajectories downsampled to model time units.

        """
        traj = []
        traj.append(init_state)
        active_traj = np.arange(N_traj) # Index of the trajectories that are still running
        i = 0
        transitions = 0
        transit_back = 0
        while len(active_traj) > 0:
            noise_term = self.noise_factor * self.rng.normal(loc=0.0, scale=np.sqrt(self.dt), size=len(active_traj))
            t_current, x_current = traj[i][active_traj, 0], traj[i][active_traj, 1]

            t_new, x_new = np.full(N_traj, np.nan), np.full(N_traj, np.nan)
            t_new[active_traj], x_new[active_traj] = self.euler_maruyama(t_current, x_current, self.dt, self.mu, noise_term)
            traj.append(np.stack([t_new, x_new], axis=1))

            back_to_on = ~self.is_on(traj[i][active_traj]) & self.is_on(traj[i+1][active_traj])
            reached_off = self.is_off(traj[i+1][active_traj])
            if np.any(reached_off):
                transitions += np.sum(reached_off)
            if np.any(back_to_on):
                transit_back += np.sum(back_to_on)
            active_traj = active_traj[np.flatnonzero(~(back_to_on | reached_off))]  
            i += 1
        
        traj = np.array(traj)
        traj = np.transpose(traj, (1, 0, 2))
        prob = transitions/N_traj
        #print('Number of trajectories:', traj.shape[0])
        #print('Number of simulated timsteps:', traj.shape[1])
        #print('Transitioned trajectories:', transitions)
        #print('Transited back trajectories:', transit_back)
        #print(f'Probability: {prob:.3f}')

        if downsample==True:
            # Downsample the trajectory to return model time units
            i = int(1 / self.dt)
            traj = traj[:, ::i, :]

        return traj, prob


    def get_pullback(self, return_between_equil: bool = False, N_traj=100, T_max=400, t_0=-200):
        '''
        Parameters
        ----------
        return_between_equil: boolean
            If true, returns the pullback trajectory only between t=0 and x>=1.
            Idea: This is the range where we want a score function
            Default: False

        Returns
        -------
        traj: np.array of shape (int(Tmax/dt), 2)
            The estimated pullback trajectory
        '''

        t_init = np.full(N_traj, t_0)
        x_init = np.linspace(-2, 2, N_traj)
        initial_state = np.stack([t_init, x_init], axis=1)
        traj = self.trajectory_TAMS(
            N_traj, T_max, initial_state, return_all=True, noise_factor=0
        ) # noise_factor=0 to get deterministic trajectory & return all timesteps
        traj = np.mean(traj, axis=0)
        if return_between_equil==True:
            mask = (traj[:,0] >= 0) & (traj[:,1] <= 1.0)
            traj = traj[mask]
        self.PB_traj = traj
        return traj

   

if __name__ == "__main__":

    mu = 0.03
    noise_factor = 0.1
    DW_model = DoubleWell_1D(mu, noise_factor)
    times = np.arange(0, 25, 0.01, dtype=float).round(2)
    DW_model.set_roots(times)
    

    N_traj = 1000
    init_state = np.array([[5.0, -0.2]])
    init_state = np.tile(init_state, (N_traj,1))

    start = time.perf_counter()
    traj, prob = DW_model.trajectory_AMS(N_traj, init_state, downsample=False)
    simulated_steps = traj.shape[1]
    end = time.perf_counter()
    print(f'Time per step:{((end-start)/simulated_steps):.6f} seconds')
    print(f'Probability: {prob:.3f}')

'''
    roots_off = np.vectorize(DW_model.off_dict.get)(times)
    roots_on = np.vectorize(DW_model.on_dict.get)(times)
    fig, ax = plt.subplots(dpi=250)
    for i in range(N_traj):
        ax.plot(traj[i, :, 0], traj[i, :, 1], linewidth=0.8)
    ax.plot(times, roots_off, label='Off-state')
    ax.plot(times, roots_on, label='On-state')
    ax.scatter(init_state[0,0], init_state[0,1], color='black', label='Initial State', s=30, zorder=10)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    ax.legend()
    ax.grid()
    ax.set_title(f"{N_traj} Trajectories with mu={mu} & noise = {noise_factor}")
    fig.savefig('../temp/trajectories.png')
    fig, ax = plt.subplots(dpi=250)
    ax = DW_model.plot_potential(5, ax)
    ax = DW_model.plot_potential(10, ax)
    ax = DW_model.plot_potential(15, ax)
    ax = DW_model.plot_potential(20, ax)
    ax.legend()
    ax.grid()
    ax.set_title(f"Potentials for mu={mu}")
    fig.savefig('../temp/potentials.png')
'''




""" mu = 0.03
    noise_factor = 0.05
    model = DoubleWell_1D(mu, noise_factor)
    N_traj = 10000
    initial_times = np.arange(0, 4, 0.1, dtype=float)
    initial_positions = np.arange(-1.0, -0.7, 0.002, dtype=float)
    print(initial_times, initial_positions)
    T, P = np.meshgrid(initial_times, initial_positions)
    filepath = f'../temp/simulation_grid{T.shape[0]*T.shape[1]}_noise_5e2.txt'
    with tqdm(total=T.shape[0] * T.shape[1]) as pbar:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                init_state = np.array([T[i,j],P[i,j]])
                init_state = np.tile(init_state, (N_traj,1))
                _, prob = model.trajectory_AMS(N_traj, init_state, downsample=False)
                with open(filepath, "a") as f:
                    f.write(
                        f"{T[i,j]};{P[i,j]};{prob} \n"
                    )
                pbar.update(1)
    """