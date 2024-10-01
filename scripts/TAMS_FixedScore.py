# -*- coding: utf-8 -*-


import numpy as np



class TAMS_prescribed():
    """
    Class implementing the TAMS algorithm for prescribed score functions.
    """

    def __init__(self, N_traj, Tmax, model, scorefunc, params=None, seed=None):
        """
        Parameters
        ----------
        N : int
            Number of trajectories.
        Tmax : int
            Maximum length of trajectories.
        model : class
            Model to use.
        scorefunc : class
            Score function to use.
        params : dict, optional
            Parameters of the model.
        seed : int, optional
            Seed for the random number generator.
        """
        self.model, self.scorefunc = model, scorefunc
        self.params = params
        self.N, self.Tmax = N_traj, Tmax
        self.rng = np.random.default_rng(seed=seed)
        
    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state
    
    def run(self, ic, nc=10):
        """
        The TAMS algorithm.
        
        Parameters
        ----------
        ic : list
            Initial condition for each trajectory. It is a list of shape (2).
        nc : int, optional
            Number of clones deleted at each iteration. The default is 10.
        tot : bool, optional
            Whether we study S-transitions or F-transitions. It is only used within the model class.
            The default is False (F-transitions).
        """

        # Initial weights
        k, w = 0, 1
        
        #Turn the initial condition into Nx2 array
        ic = np.tile(ic, (self.N, 1))


        # Initial pool of trajectories
        traj = self.model.trajectory(self.N, self.Tmax, *self.params.values(),
                                       init_state=ic)
        
        # Initial number of timesteps
        Nt = traj.shape[1]
        nb_total_timesteps = Nt*self.N
        
        # Initial score
        # Set to 0 for the on-states, to 1 for the off-states
        score = self.scorefunc.process(traj)
        onzone = self.model.is_on(traj)
        offzone = self.model.is_off(traj)
        score[onzone], score[offzone] = 0, 1
        
        # Maximum score for each trajectory
        Q = np.nanmax(score,axis=1)
        
        # Loop until you cannot discard nc trajectories anymore
        while len(np.unique(Q))>nc:

            print('Q: ',Q)
            print('Number of unique maximum scores:',len(np.unique(Q)))
            
            # Find the nc trajectories with the lowest score
            threshold = np.unique(Q)[nc-1] #Because Python counts from 0
            print('Threshold to keep trajectory',threshold)
            idx, other_idx = np.nonzero(Q<=threshold)[0], np.nonzero(Q>threshold)[0]
            print('Number of trajectories to discard',len(idx))
            print('Number of trajectories to keep',len(other_idx))
            Q_min = Q[idx]
            
            #Update weights
            w *= (1-len(idx)/self.N)
            
            # Create new trajectories
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.argmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart]
            
            # Update the number of timesteps
            all_length = Nt - restart
            nb_total_timesteps += np.sum(all_length)
            
            # Create new trajectories
            new_traj = self.model.trajectory(len(idx), int(self.Tmax*np.max(all_length)/Nt), *self.params.values(),
                                             init_state=init_clone)

            # Update the score
            new_score = self.scorefunc.process(new_traj)
            
            # Update the trajectories and the score
            for i in range(len(idx)):
                t, r, l = idx[i], restart[i], all_length[i]
                
                traj[t,:r] = traj[new_ind[i],:r]
                traj[t,r:] = new_traj[i,:l]
                
                score[t,:r+1] = score[new_ind[i],:r+1]
                score[t,r+1:] = new_score[i,1:l]
                
                onzone = self.model.is_on(traj[t])
                offzone = self.model.is_off(traj[t])

                score[t, onzone], score[t, offzone] = 0, 1
                
                # If some trajectories have hit the transition, they end with NaNs
                first_idx_off = np.argmax(offzone)
                if first_idx_off>0:
                    score[t,first_idx_off+1:] = np.nan
                    traj[t,first_idx_off+1:] = np.nan
                
            k += 1
            
            Q = np.nanmax(score,axis=1)

        # Return the probability of the transition, the number of iterations, the number of trajectories that transitioned,
        #        the number of timesteps and the trajectories
        return w*np.count_nonzero(Q>=1)/self.N, k, np.count_nonzero(Q>=1), nb_total_timesteps, traj
    
if __name__ == "__main__":
    
    from DoubleWell_Model import DoubleWell_1D
    from Score_Function import x_coord
    from tqdm import trange

    # Various TAMS parameters
    nc = 1 #number of discarded per iteration
    nb_runs=1 #number of full TAMS runs
    N = 10 #number of trajectories per run
    tmax = 100 #maximum length of trajectories in model time units
    dt=0.01 #time step of the simulation in model time units
    mu=0 #coupling parameter in the model
    noise_factor=0.1 #noise parameter in the model
    initial_condition = [0.0,-1.0] #initial condition for [time,x]


    C_model=DoubleWell_1D(on_state = -1.0)
    scorefunc = x_coord(C_model,init_state=initial_condition)
    tams = TAMS_prescribed(N, tmax, C_model, scorefunc, params={"dt":dt,"mu":mu,"noise":noise_factor})



    rng_model = np.random.default_rng(seed=23)
    seeds_model = rng_model.choice(100, size=nb_runs, replace=False)

    ### RUN TAMS FOR THE DOUBLE WELL MODEL
    for r in range(nb_runs):
        C_model.rng.bit_generator.state = np.random.PCG64(seeds_model[r]).state
        probability, tams_iterations, transition, timesteps, trajectories = tams.run(initial_condition, nc=nc)
        print("Probability of transition: ", probability)
        print("Number of iterations: ", tams_iterations)
        print("Number of trajectories that transitioned: ", transition)
        print("Number of timesteps: ", timesteps)
 
    


        