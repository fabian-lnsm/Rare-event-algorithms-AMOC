
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from tqdm import tqdm





class AMS():
    
    def __init__(self, N_traj, nc, seed=None):
        self.N_traj, self.nc = N_traj, nc
        self.rng = np.random.default_rng(seed=seed)
        self.dimension = 2 # includes time

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state


    def set_score(self, score_function, *fixed_args, **fixed_kwargs):
            self.score_function = score_function
            self.fixed_args = fixed_args
            self.fixed_kwargs = fixed_kwargs

    def comp_score(self, traj):
        return self.score_function(traj, *self.fixed_args, **self.fixed_kwargs)


    def set_model(self, model):
        self.model = model

    def set_traj_func(self, traj_function, *fixed_args, **fixed_kwargs):
        self.traj_function = traj_function  # Store the function
        self.traj_fixed_args = fixed_args  # Store fixed args
        self.traj_fixed_kwargs = fixed_kwargs  # Store fixed kwargs

    def comp_traj(self, N_traj : int, init_state : np.array):
        '''
        Parameters
        ----------
        N_traj : int
            Number of trajectories to generate
        init_state: np.array of shape (N_traj, dimension)
                Initial state of the trajectories

        Returns
        -------
        np.array of shape (N_traj, time, dimension)
            The simulated Trajectories
        '''

        return self.traj_function(N_traj, init_state, *self.traj_fixed_args, **self.traj_fixed_kwargs)

    def run(self, init_state, zmax=1):
        '''
        Parameters
        ----------
        init_state : np.array of shape (N_traj, dimension)
            Initial state of the trajectories
        zmax : float. Default is 1
            Threshold for collapse = maximum score

        Returns
        -------
        dict
            A dictionary containing the following keys:
            
            - **probability** (_float_):  
            Probability of collapse.
            
            - **iterations** (_int_):  
            Number of iterations.
            
            - **nb_transitions** (_int_):  
            Number of transitions.
            
            - **trajectories** (_np.array of shape (N_traj, time, dimension)_):  
            Trajectories. Time is not fixed.

            - **scores** (_np.array of shape (N_traj, time)_):
            Scores of the trajectories.

            - **runtime** (_float_):
            Execution time.
            
        '''
        k, w = 0, 1

        # For convenience, trajectories are stored in an array of shape (N, T, dimension)
        # But trajectories stop when they reach either region B or come back to region A
        # So they all have different lengths
        # All trajectories are padded at the end with NaNs so that they have the same length

        traj = self.comp_traj(self.N_traj, init_state)

        # Find the actual length of each trajectory (i.e. the first NaN)
        Nt = np.argmax(np.isnan(traj).any(axis=2), axis=1)
        Nt = np.where(Nt == 0, traj.shape[1], Nt) # If there is no NaN, it is the longest trajectory
        max_Nt = np.max(Nt) 

        score = self.comp_score(traj)
        onzone = self.model.is_on(traj)
        offzone = self.model.is_off(traj)
        score[onzone], score[offzone] = 0, 1

        Q = np.nanmax(score,axis=1)

        while len(np.unique(Q)) > self.nc:

            threshold = np.unique(Q)[self.nc-1] #Because Python counts from 0
            idx, other_idx = np.flatnonzero(Q<=threshold), np.flatnonzero(Q>threshold)
            Q_min = Q[idx]

            #Update weights
            w *= (1-len(idx)/self.N_traj)

            # Create new trajectories
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.nanargmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart,:]

            new_traj = self.comp_traj(len(idx), init_clone)
            new_score = self.comp_score(new_traj)


            # Find the actual length of each new trajectory (i.e. the first NaN)
            new_nt = np.argmax(np.isnan(new_traj).any(axis=2), axis=1)
            new_nt[np.all(~np.isnan(new_traj), axis=(1, 2))] = new_traj.shape[1]

            # Update the length of the resimulated trajectories
            Nt[idx] = restart + new_nt
            new_max = np.max(Nt[idx])

            # If the nex max length is greater than the current max length, we need to pad the arrays
            # We need to add new_max - max_Nt NaNs at the end of each trajectory to make sure the array traj is large enough
            if new_max > max_Nt:
                if self.dimension > 1:
                    traj = np.concatenate((traj, np.full((self.N_traj,new_max-max_Nt,self.dimension),np.nan)), axis=1)
                else:
                    traj = np.concatenate((traj, np.full((self.N_traj,new_max-max_Nt),np.nan)), axis=1)
                score = np.concatenate((score, np.full((self.N_traj,new_max-max_Nt),np.nan)), axis=1)
                max_Nt = new_max

            # update the trajectories and scores
            for i in range(len(idx)):
                t, r, l = idx[i], restart[i], new_nt[i]

                traj[t,:r+1,:] = traj[new_ind[i],:r+1,:]
                traj[t,r+1:r+l,:] = new_traj[i,1:l,:]

                score[t,:r+1] = score[new_ind[i],:r+1]
                score[t,r+1:r+l] = new_score[i,1:l]

                onzone = self.model.is_on(traj[t])
                offzone = self.model.is_off(traj[t])

                score[t, onzone], score[t, offzone] = 0, 1

            #Prepare next iteration
            k += 1
            Q = np.nanmax(score,axis=1)

        count_collapse = np.count_nonzero(Q>=zmax)

        return dict({
            'probability':w*count_collapse/self.N_traj,
            'iterations':k,
            'nb_transitions':count_collapse,
            'trajectories':traj,
            'scores':score,
            })
    
    def _run_single(self, init_state):
        # This function is intended to be called by each worker process.
        return self.run(init_state)
    
    def run_multiple(self, nb_runs:int, init_state : np.array):
        '''
        Parameters
        ----------
        nb_runs : int
            Number of runs
        init_state : np.array of shape (N_traj, dimension) or (dimension,)
            Initial state of all trajectories. \n
            If the dimension is 1, the initial state is copied to be the same for each of the N_traj trajectories.

        Returns
        -------
        np.array of shape (nb_runs, 3)
            Array containing the results of the runs. Each row contains the following information:
            - Probability
            - Number of iterations
            - Number of transitions

        '''
        #print(f'Running {nb_runs} simulations...', flush=True)

        if init_state.ndim == 1:
            init_state = np.tile(init_state, (self.N_traj,1))

        t_start = time.perf_counter()
        with Pool() as pool:
            # Map the run method across nb_runs
            results = pool.map(self._run_single, [init_state] * nb_runs)

        stats = np.zeros((nb_runs, 3))
        for i, result in enumerate(results):
            stats[i,0] = result['probability']
            stats[i,1] = result['iterations']
            stats[i,2] = result['nb_transitions']

        t_end = time.perf_counter()
        runtime = t_end - t_start
        #print('runtime:', runtime, flush=True)
        return np.mean(stats[:,0]), np.std(stats[:,0])

        
if __name__ == "__main__":

    from DoubleWell_Model import DoubleWell_1D
    from Score_Function import score_x, score_PB


    mu = 0.03
    dt = 0.01
    model = DoubleWell_1D(mu, dt=dt)
    score_fct = score_x()

    N_traj = 1000
    nc = 10
    AMS_algorithm = AMS(N_traj, nc)
    AMS_algorithm.set_score(score_fct.get_score)
    AMS_algorithm.set_model(model)
    AMS_algorithm.set_traj_func(model.trajectory_AMS, downsample=False)

    nb_runs = 5
    
    initial_times = np.arange(0,6,0.2, dtype=float)
    initial_positions = np.arange(-1,0,0.025, dtype=float)
    filepath = '../temp/'
    filename = f'simulationAMS_runs{nb_runs}_grid{initial_times.shape[0] * initial_positions.shape[0]}.txt'
    print(initial_times, initial_positions)
    T, P = np.meshgrid(initial_times, initial_positions)
    with tqdm(total=T.shape[0] * T.shape[1]) as pbar:
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                init_state = np.array([T[i,j],P[i,j]])
                a, b = AMS_algorithm.run_multiple(nb_runs,init_state)
                with open(filepath + filename, "a") as f:
                    f.write(
                        f"{T[i,j]};{P[i,j]};{a};{b} \n"
                    )
                pbar.update(1)



   



