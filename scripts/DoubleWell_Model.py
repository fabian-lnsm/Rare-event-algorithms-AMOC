

import numpy as np


class DoubleWell_1D():
    """
    Class implementing the Double-Well model in 1D.
    """
    
    def __init__(self, on_state=-1.0, seed=None):

        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.on = on_state
        self.off= -on_state

        

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
        distance_to_on = traj[...,1]-self.on
        return (distance_to_on <= 0) 

        
    def is_off(self, traj):
        """
        Checks for off-states in a set of trajectories.
        Off-state = right equilibrium point of the bistable system

        Parameters
        ----------
        traj : shape (Number of trajectories, timesteps, 6)
            TRAJ MUST BE NON-DIMENSIONALIZED
            
        Returns
        -------
        An array of shape (Number of trajectories, timesteps)
        Every off-state is replaced with 1, all other states are replaced with 0

        """
        distance_to_off = traj[...,1]-self.off
        return (distance_to_off >= 0)
    
    def potential(self, t,x,mu):
        """
        The potential of the system at a given state.
        """
        return x**4/4 - x**2/2 - mu*x*t
    

    def force(self, t, x, mu):
        """
        The force of the system at a given state.
        """
        return -x**3 + x + mu * t
    
    
    def euler_maruyama(self, t, x, dt, mu, noise_factor):
        noise_term = self.rng.normal(loc=0.0, scale=np.sqrt(dt))
        t_new = t + dt
        x_new = x + self.force(t, x, mu) * dt + noise_factor * noise_term
        return t_new, x_new
    

    def run_simulation(self, T_max, dt, mu, noise_factor, init_state):
        n_steps = int(T_max / dt)
        trajectory = np.zeros((n_steps,2))
        t = init_state[0]
        x = init_state[1]
        for i in range(n_steps):
            trajectory[i,:] = [t, x]
            t, x = self.euler_maruyama(t, x, dt, mu, noise_factor)
        # Downsample the trajectory to model time units
        i=int(1/dt)
        trajectory = trajectory[::i,:]
        return trajectory


    def trajectory(self, N_traj, T_max, dt, mu, noise_factor, init_state):  
        """
        Compute trajectories of the system of fixed length.

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
            The initial conditions of the given trajectories.

        Returns
        -------
        simulated_trajectories : traj of shape (N_traj, T_max, 2)
            The computed trajectories.

        """

        # check if our model is valid
        assert mu*T_max <= 0.25, "Time dependence of V(t) too strong: Would need different model"

        simulated_trajectories = np.zeros((N_traj, T_max, 2))
        
        for i in range(N_traj):
            simulated_trajectories[i,:,:] = self.run_simulation(T_max, dt, mu, noise_factor, init_state[i,:])

        return simulated_trajectories
    
    
