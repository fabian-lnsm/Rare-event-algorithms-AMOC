
import numpy as np
import matplotlib.pyplot as plt

from DoubleWell_Model import DoubleWell_1D


class score_x:
    """
    Assumes that the equilibrium states are at x=1 and x=-1
    """

    def __init__(self):
        self.equilibrium = 1  # abs(x-coordinate) of the equilibrium states

    def get_score(self, traj : np.array):
        """
        Parameters
        ----------
        traj: np.array of shape (...,2)

        Returns
        -------
        score: np.array of shape (...)
        """
        x_value = traj[..., 1]
        score = (x_value + self.equilibrium) / (2 * self.equilibrium)
        return score



class AMS():
    
    def __init__(self, N_traj, nc, seed=None):
        self.N_traj, self.nc = N_traj, nc
        self.rng = np.random.default_rng(seed=seed)
        self.dimension = 2

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state

    def set_score(self, score_function, *fixed_args, **fixed_kwargs):
        self.comp_score = lambda traj : score_function(traj, *fixed_args, **fixed_kwargs)

    def set_traj_func(self, traj_function, *fixed_args, **fixed_kwargs):
        '''
        Method
        ------
        Set the trajectory function. \n
        Signature should be:
            traj_function(N_traj, init_state, *fixed_args, **fixed_kwargs)
        where:
            - **N_traj** (_int_)
                Number of trajectories to generate
            - **init_state** (_np.array of shape (N_traj, dimension)_)
                Initial state of the trajectories

        Parameters
        ----------
        traj_function : function
            Function that generates the trajectories
        fixed_args : list
            Fixed arguments of the function
        fixed_kwargs : dict
            Fixed keyword arguments of the function
        
        Returns
        -------
        None
        '''
        self.comp_traj = lambda N_traj, init_state : traj_function(N_traj, init_state, *fixed_args, **fixed_kwargs)

    def run(self, init_state, zmax=1):
        '''
        Parameters
        ----------
        init_state : np.array of shape (N_traj, dimension)
            Initial state of the trajectories
        zmax : float. Default is 1

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
            
        '''
        k, w = 0, 1

        # For convenience, trajectories are stored in an array of shape (N, T, dimension)
        # But trajectories stop when they reach either region B or come back to region A
        # So they all have different lengths
        # All trajectories are padded at the end with NaNs so that they have the same length

        traj = self.comp_traj(self.N_traj, init_state)

        # Find the actual length of each trajectory (i.e. the first NaN)
        Nt = np.argmax(np.isnan(traj), axis=1)
        Nt[Nt==0] = traj.shape[1] # If there is no NaN, it is the longest trajectory
        max_Nt = np.max(Nt) 

        score = self.comp_score(traj)

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
            init_clone = traj[new_ind,restart]

            new_traj = self.comp_traj(len(idx), init_clone)
            new_score = self.comp_score(new_traj)

            # Find the actual length of each new trajectory (i.e. the first NaN)
            new_nt = np.argmax(np.isnan(new_traj).any(axis=2), axis=1)
            new_nt[np.all(~np.isnan(new_traj), axis=(1, 2))] = new_traj.shape[1]

            # Update the length of the resimulated trajectories
            print('Nt:', Nt)
            print('idx:', idx)
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
                max_Nt = np.max(Nt[idx])

            for i in range(len(idx)):
                t, r, l = idx[i], restart[i], new_nt[i]

                traj[t,:r+1] = traj[new_ind[i],:r+1]
                traj[t,r+1:r+l] = new_traj[i,1:l]

                score[t,:r+1] = score[new_ind[i],:r+1]
                score[t,r+1:r+l] = new_score[i,1:l]

            #Prepare next iteration
            k += 1
            Q = np.nanmax(score,axis=1)

        count_collapse = np.count_nonzero(Q>=zmax)

        return dict({'probability':w*count_collapse/self.N_traj,
                     'iterations':k,
                     'nb_transitions':count_collapse,
                     'trajectories':traj,
                     'scores':score})

if __name__ == "__main__":

    mu = 0.03
    model = DoubleWell_1D(mu)

    N_traj = 10
    nc = 1
    AMS_algorithm = AMS(N_traj, nc)
    AMS_algorithm.set_score(score_x().get_score)
    AMS_algorithm.set_traj_func(model.trajectory_AMS, downsample=False)

    init_state = np.tile(np.array([0.0,-1.0]), (N_traj,1))
    res = AMS_algorithm.run(init_state)
    print(res)



