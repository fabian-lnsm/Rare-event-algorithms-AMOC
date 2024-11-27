import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time as time
from multiprocessing import Pool


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

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state    
    
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
        self.root_times = all_t
        self.on_dict = dict(zip(all_t.T, roots[:, 1])) #left equilibrium point
        self.off_dict = dict(zip(all_t.T, roots[:, 0])) #right equilibrium point

    def plot_OnOff(self, times, ax):
        self.set_roots(times)
        off_state = np.vectorize(self.off_dict.get)(times)
        on_state = np.vectorize(self.on_dict.get)(times)
        ax.plot(times, off_state, label='On/Off-states', color='blue', linestyle='--')
        ax.plot(times, on_state, color='blue', linestyle='--')
        return ax

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
    
    
    def simulate_AMS_MC_multiple(
        self,
        init_state: np.ndarray,
        nb_runs: int = 1,
        N_transitions: int = 30,
        N_traj: int = 100000,
        filepath: str = None,
    ):
        '''
        Runs MC simulation multiple times for a given initial state. Returns mean and std of the transition probability.
        '''
        # reset class properties and print some information
        self.nb_runs = nb_runs
        self.init_state = init_state[0,:]
        print('-'*50, flush=True)
        print(f'Initial state: {self.init_state}', flush=True)
        print(f'Number of runs: {self.nb_runs}', flush=True)
        print('-'*50, flush=True)

        # Run the simulations
        t_start = time.perf_counter()
        stats = np.zeros((nb_runs, 3))
        seeds = [np.random.randint(0, 2**16 - 1) for _ in range(nb_runs)]
        for i, seed in enumerate(seeds):
            print(f'Run {i+1}/{nb_runs}', flush=True)
            self.reset_seed(seed)
            result = self.simulate_AMS_MC(init_state, N_transitions, N_traj, printing=False)
            stats[i,0] = result['probability']
            stats[i,1] = result['simulated_traj']
            stats[i,2] = result['transitions']            
        t_end = time.perf_counter()
        runtime = t_end - t_start
        print('Total runtime:', runtime, flush=True)
        print(f'Probability: {np.mean(stats[:,0])} +/- {np.std(stats[:,0])}', flush=True)

        # save results to file
        if filepath is not None:
            prob = (np.mean(stats[:,0]), np.std(stats[:,0], ddof=1))
            simulated_traj = (np.mean(stats[:,1]), np.std(stats[:,1], ddof=1))
            transitions = (np.mean(stats[:,2]), np.std(stats[:,2], ddof=1))
            with open(filepath, 'a') as f:
                f.write(f'Model: g={self.noise_factor}, mu={self.mu}, runs={self.nb_runs}\n')
                f.write(f'Runtime: {runtime}\n')
                f.write(f'Initial state: {self.init_state}\n')
                f.write(f'Probability: {prob[0]} +/- {prob[1]}\n')
                f.write(f'Simulated traj: {simulated_traj[0]} +/- {simulated_traj[1]}\n')
                f.write(f'Transitions: {transitions[0]} +/- {transitions[1]}\n')
                f.write('\n')

        return stats


    def simulate_AMS_MC(
        self,
        init_state: np.ndarray,
        N_transitions: int = 30,
        N_traj: int = 100000,
        printing: bool = False,
    ):
        '''
        Compute Trajectories until a fixed number of transitions is reached. Returns the transition probability.
        '''
        if printing==True:
            print('-'*50)
            print(f'Number of transitions: {N_transitions}', flush=True)
            print(f'Number of trajectories per run: {N_traj}', flush=True)

        transitions = 0
        simulated_traj = 0
        time_start = time.perf_counter()
        while transitions < N_transitions:
            _, _, transit = self.trajectory_AMS(N_traj, init_state, downsample=False)
            transitions += transit
            simulated_traj += N_traj
            print(f'Current: {simulated_traj} traj with {transitions} transitions', flush=True)
        prob = transitions/simulated_traj
        time_end = time.perf_counter()
        runtime = time_end - time_start

        if printing==True:
            print('-'*50)
            print('Success!')
            print('-'*50)
            print(f'Total runtime: {runtime}')
            print(f'Number of trajectories: {simulated_traj}')
            print(f'Number of transitions: {transitions}')
            print(f'Probability: {prob}')
        
        return dict ({'probability': prob, 'simulated_traj': simulated_traj, 'transitions': transitions})
        
    
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

        return traj, prob, transitions


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
            mask = (traj[:,0] >= 0) & (traj[:,1] <= 1.3)
            traj = traj[mask]
        self.PB_traj = traj
        return traj

    def plot_OnOffStates(self,  ax):
        '''
        Plot the on/off states of the system for a given time interval.
        '''
        off_state = np.vectorize(self.off_dict.get)(self.root_times)
        on_state = np.vectorize(self.on_dict.get)(self.root_times)
        ax.plot(self.root_times, off_state, label='Off-state', color='darkred')
        ax.plot(self.root_times, on_state, label='On-state', color='blue')
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Position $x$")
        ax.grid()
        ax.set_title(f"Phase space with mu={self.mu} & noise = {self.noise_factor}")
        return ax

    def plot_pullback(self, ax):
        '''
        Plot the pullback trajectory of the system.
        '''
        PB_traj = self.get_pullback(return_between_equil = True)
        ax.plot(
                PB_traj[:, 0],
                PB_traj[:, 1],
                label="PB attractor",
                color="black", linewidth=2, linestyle='--'
                )
        return ax

if __name__ == "__main__":

    # Set up the model
    mu = 0.03
    noise_factor = 0.1
    DW_model = DoubleWell_1D(mu, noise_factor)
    root_times = np.arange(0, 30, 0.01, dtype=float).round(2)  # Time points for which the roots should be computed
    DW_model.set_roots(root_times) 

    # Set up initial states
    init_times = np.array([4.0, 7.0, 10.0, 2.0])
    init_positions = np.vectorize(DW_model.on_dict.get)(init_times)
    init_states = np.stack([init_times, init_positions], axis=1)
    print(init_states)

    #Plotting phase-space with initial states: Writes plot to file
    fig, ax = plt.subplots(dpi=250)
    ax = DW_model.plot_OnOffStates(ax)
    ax.scatter(init_states[:,0], init_states[:,1], color='black', label='Initial States', s=30, zorder=10)
    ax = DW_model.plot_pullback(ax)
    ax.legend()
    fig.savefig('../temp/phase_space.png')
    
    # Run MC multiple times for each initial state
    nb_runs = 20
    N_traj = 100000
    for init_state in init_states:
        init_state = np.tile(init_state, (N_traj,1))
        stats = DW_model.simulate_AMS_MC_multiple(init_state, nb_runs, N_traj = N_traj, N_transitions=30, filepath='../temp/MC_results.txt')
    
