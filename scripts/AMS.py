
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

    def set_modelroots(self, times=None):
        self.model.set_roots(times)

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
        traj, _ , _ = self.traj_function(N_traj, init_state, *self.traj_fixed_args, **self.traj_fixed_kwargs)
        return traj
    
    def get_true_length(self, traj):
        return np.where(np.isnan(traj).any(axis=2).any(axis=1), np.argmax(np.isnan(traj).any(axis=2), axis=1), traj.shape[1])

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


            
        '''
        k, w = 0, 1

        traj = self.comp_traj(self.N_traj, init_state)
        max_length = traj.shape[1]
        score = self.comp_score(traj)
        #onzone = self.model.is_on(traj)
        offzone = self.model.is_off(traj)
        #score[onzone], score[offzone] = 0, 1
        score[offzone] = 1

        Q = np.nanmax(score, axis=1)

        while len(np.unique(Q)) > self.nc:
            threshold = np.unique(Q)[self.nc-1]
            idx, other_idx = np.flatnonzero(Q<=threshold), np.flatnonzero(Q>threshold)
            w *= (1-len(idx)/self.N_traj)
            Q_min = Q[idx]

            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.nanargmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart,:]
            new_traj = self.comp_traj(len(idx), init_clone)
            max_length_newtraj = np.max(restart + self.get_true_length(new_traj))

            if max_length_newtraj > max_length:
                traj = np.concatenate((traj, np.full((self.N_traj,max_length_newtraj-max_length,self.dimension),np.nan)), axis=1)
                score = np.concatenate((score, np.full((self.N_traj,max_length_newtraj-max_length),np.nan)), axis=1)
                max_length = max_length_newtraj

            for i in range(len(idx)):
                tr_idx, rs, length = idx[i], restart[i], self.get_true_length(new_traj)[i]
                traj[tr_idx,:rs+1,:] = traj[new_ind[i],:rs+1,:]
                traj[tr_idx,rs+1:rs+length,:] = new_traj[i,1:length,:]
                traj[tr_idx,rs+length:,:] = np.nan
                score[tr_idx,:] = self.comp_score(traj[tr_idx,:,:])
                #onzone = self.model.is_on(traj[tr_idx])
                offzone = self.model.is_off(traj[tr_idx])
                #score[tr_idx, onzone], score[tr_idx, offzone] = 0, 1
                score[tr_idx, offzone] = 1

                    

            #Prepare next iteration
            k += 1
            if k % 100 == 0:
                print('Iteration:', k, 'Number of transitions:', np.count_nonzero(Q>=zmax), flush=True)
            if k % 10 == 0:
                self.plot_trajectories_during_run(traj, k)
            Q = np.nanmax(score,axis=1)

        count_collapse = np.count_nonzero(Q>=zmax)
        return dict({
            'probability':w*count_collapse/self.N_traj,
            'iterations':k,
            'nb_transitions':count_collapse,
            'trajectories':traj,
            'scores':score,
            })
    
    def plot_trajectories_during_run(self, traj, k):
        fig, ax = plt.subplots(dpi=250)
        for i in range(self.N_traj):
            ax.plot(traj[i,:,0], traj[i,:,1])
        ax.set_title(f'AMS iteration {k}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Position x')
        fig.savefig(f'../temp/traj/AMS_{k}.png')
        plt.close(fig)
    
    def _run_single(self, init_state, seed):
        self.reset_seed(seed)
        return self.run(init_state)
    
    def run_multiple(self, nb_runs:int, init_state : np.array):
        '''
        Parameters
        ----------
        nb_runs : int
            Number of runs
        init_state : np.array of shape (N_traj, dimension)
            Initial state of all trajectories. \n

        Returns
        -------
        stats : np.array of shape (nb_runs, 3)

        '''
        # reset class properties and print some information
        self.init_state = init_state[0,:]
        self.nb_runs = nb_runs
        print('-'*20, flush=True)
        print('-'*20, flush=True)
        print(f'Initial state: {self.init_state}', flush=True)
        print(f'Running {nb_runs} simulations...', flush=True)

        # Run the simulations
        t_start = time.perf_counter()
        seeds = [np.random.randint(0, 2**16 - 1) for _ in range(nb_runs)]
        with Pool() as pool:
            results = pool.starmap(self._run_single, [(init_state, seed) for seed in seeds])

        stats = np.zeros((nb_runs, 3))
        for i, result in enumerate(results):
            stats[i,0] = result['probability']
            stats[i,1] = result['iterations']
            stats[i,2] = result['nb_transitions']

        # print and return results
        t_end = time.perf_counter()
        runtime = t_end - t_start
        self.runtime = runtime
        print('Total runtime:', runtime, flush=True)
        print(f'Probability: {np.mean(stats[:,0])} +/- {np.std(stats[:,0])}', flush=True)
        return stats
    
    def write_results(self, stats, filename):
        prob = (np.mean(stats[:,0]), np.std(stats[:,0], ddof=1))
        iter = (np.mean(stats[:,1]), np.std(stats[:,1], ddof=1))
        trans = (np.mean(stats[:,2]), np.std(stats[:,2], ddof=1))
        with open(filename, 'a') as f:
            f.write(f'Model: g={self.model.noise_factor}, mu={self.model.mu}, runs={self.nb_runs}, N_traj={self.N_traj}, nc={self.nc} \n')
            f.write(f'Score function: {self.score_function}\n')
            f.write(f'Runtime: {self.runtime}\n')
            f.write(f'Initial state: {self.init_state}\n')
            f.write(f'Probability: {prob[0]} +/- {prob[1]}\n')
            f.write(f'Iterations: {iter[0]} +/- {iter[1]}\n')
            f.write(f'Transitions: {trans[0]} +/- {trans[1]}\n')
            f.write('\n')

        
if __name__ == "__main__":

    # Import necessary modules
    from DoubleWell_Model import DoubleWell_1D
    from Score_Function import score_x, score_PB

    #model parameters
    mu = 0.03
    dt = 0.01
    noise_factor = 0.1
    DW_model = DoubleWell_1D(mu, dt=dt, noise_factor=noise_factor)
    #score_fct = score_PB(DW_model)
    score_fct = score_x()

    #AMS parameters
    N_traj = 100
    nc = 1
    nb_runs = 1

    # Initialize AMS algorithm
    AMS_algorithm = AMS(N_traj, nc)
    AMS_algorithm.set_score(score_fct.get_score)
    AMS_algorithm.set_model(DW_model)
    AMS_algorithm.set_traj_func(DW_model.trajectory_AMS, downsample=False)
    AMS_algorithm.set_modelroots()

    # Create Initial states
    #init_times = np.array([2.0, 4.0, 7.0, 10.0])
    init_times = np.array([4.0])
    init_positions = np.vectorize(DW_model.on_dict.get)(init_times)
    init_states = np.stack([init_times, init_positions], axis=1)
    print('Init states: ',init_states)

    # Run AMS algorithm multiple times for each initial state and write results
    for _, init_state in enumerate(init_states):
        init_state = np.tile(init_state, (N_traj,1))
        results = AMS_algorithm.run_multiple(nb_runs, init_state)
        AMS_algorithm.write_results(results, '../temp/AMS_results.txt')


   
    



   



