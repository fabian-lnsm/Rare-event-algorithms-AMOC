
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
        if times is None:
            times = np.arange(0, 50, 0.01, dtype=float).round(2)
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
        traj, _ = self.traj_function(N_traj, init_state, *self.traj_fixed_args, **self.traj_fixed_kwargs)
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

            - **runtime** (_float_):
            Execution time.
            
        '''
        k, w = 0, 1

        traj = self.comp_traj(self.N_traj, init_state)
        max_length = traj.shape[1]
        score = self.comp_score(traj)
        onzone = self.model.is_on(traj)
        offzone = self.model.is_off(traj)
        score[onzone], score[offzone] = 0, 1

        Q = np.nanmax(score,axis=1)
        time_idx, time_newtraj, time_update = [], [], []
        while len(np.unique(Q)) > self.nc:
            time_start = time.perf_counter()
            threshold = np.unique(Q)[self.nc-1]
            idx, other_idx = np.flatnonzero(Q<=threshold), np.flatnonzero(Q>threshold)
            w *= (1-len(idx)/self.N_traj)
            Q_min = Q[idx]
            time_idx.append(time.perf_counter() - time_start)

            time_start = time.perf_counter()
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.nanargmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart,:]
            new_traj = self.comp_traj(len(idx), init_clone)
            max_length_newtraj = np.max(restart + self.get_true_length(new_traj))
            time_newtraj.append(time.perf_counter() - time_start)

            time_start = time.perf_counter()
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
                onzone = self.model.is_on(traj[tr_idx])
                offzone = self.model.is_off(traj[tr_idx])
                score[tr_idx, onzone], score[tr_idx, offzone] = 0, 1
            
            time_update.append(time.perf_counter() - time_start)
            

            #Prepare next iteration
            k += 1
            Q = np.nanmax(score,axis=1)


        print('time_idx:', sum(time_idx))
        print('time_newtraj:', sum(time_newtraj))
        print('time_update:', sum(time_update))
        count_collapse = np.count_nonzero(Q>=zmax)
        return dict({
            'probability':w*count_collapse/self.N_traj,
            'iterations':k,
            'nb_transitions':count_collapse,
            'trajectories':traj,
            'scores':score,
            })
    
    def _run_single(self, init_state):
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
        Probability_mean, Probability_stddev : (float, float)
            Tuple containing the result of the runs:
            - Probability: Mean
            - Probability: Standard deviation

        '''
        print(f'Running {nb_runs} simulations...', flush=True)

        if init_state.ndim == 1:
            init_state = np.tile(init_state, (self.N_traj,1))

        t_start = time.perf_counter()
        with Pool() as pool:
            results = pool.map(self._run_single, [init_state] * nb_runs)

        stats = np.zeros((nb_runs, 3))
        for i, result in enumerate(results):
            stats[i,0] = result['probability']
            stats[i,1] = result['iterations']
            stats[i,2] = result['nb_transitions']

        t_end = time.perf_counter()
        runtime = t_end - t_start
        print('runtime:', runtime, flush=True)
        return np.mean(stats[:,0]), np.std(stats[:,0])

        
if __name__ == "__main__":

    from DoubleWell_Model import DoubleWell_1D
    from Score_Function import score_x, score_PB


    mu = 0.03
    dt = 0.01
    noise_factor = 0.1
    model = DoubleWell_1D(mu, dt=dt, noise_factor=noise_factor)
    score_fct = score_x()

    N_traj = 1000
    nc = 10
    nb_runs = 5

    AMS_algorithm = AMS(N_traj, nc)
    AMS_algorithm.set_score(score_fct.get_score)
    AMS_algorithm.set_model(model)
    AMS_algorithm.set_traj_func(model.trajectory_AMS, downsample=False)
    AMS_algorithm.set_modelroots()

    init_state = np.array([[0.0, -1.0]])
    init_state = np.tile(init_state, (N_traj,1))

    result = AMS_algorithm.run(init_state)
    print('probability: ',result['probability'])

    fig, ax = plt.subplots(dpi=250)
    length_max = np.max(AMS_algorithm.get_true_length(result['trajectories']))
    times = np.arange(0, init_state[0,0] + length_max*dt, dt)
    ax = model.plot_OnOff(times, ax)
    ax.scatter(init_state[0,0], init_state[0,1], color='black', label='Initial State', s=30, zorder=10)
    for i in range(N_traj):
        ax.plot(result['trajectories'][i,:,0], result['trajectories'][i,:,1])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.legend()
    ax.grid()
    fig.savefig('../temp/AMS_trajectories.png')


    
    



   



