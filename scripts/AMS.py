
import numpy as np

class AMS():
    
    def __init__(self, N, nc, seed=None):
        self.N, self.nc = N, nc
        self.rng = np.random.default_rng(seed=seed)

    def reset_seed(self, seed):
        self.rng.bit_generator.state = np.random.PCG64(seed).state

    def set_score(self, score_function, *fixed_args, **fixed_kwargs):
        self.comp_score = lambda traj : score_function(traj, *fixed_args, **fixed_kwargs)

    def set_traj_func(self, traj_function, *fixed_args, **fixed_kwargs):
        '''
        The function computing trajectories must have the following signature:
        traj_function(n, t, ic, *args, **kwargs)
        where:
        - n is the number of trajectories to compute
        - t is the start time (important because the system is non-autonomous)
        - ic are the initial conditions
        '''
        self.comp_traj = lambda n, t, ic : traj_function(n, t, ic, *fixed_args, **fixed_kwargs)

    def run(self, ic, zmax=1):
        k, w = 0, 1

        # For convenience, trajectories are stored in an array of shape (N, T, dimension)
        # But trajectories stop when they reach either region B or come back to region A
        # So they all have different lengths
        # All trajectories are padded at the end with NaNs so that they have the same length

        traj = self.comp_traj(self.N, np.zeros(self.N), ic)

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
            w *= (1-len(idx)/self.N)

            # Create new trajectories
            new_ind = self.rng.choice(other_idx, size=len(idx))
            restart = np.nanargmax(score[new_ind]>=Q_min[:,np.newaxis], axis=1)
            init_clone = traj[new_ind,restart]

            new_traj = self.comp_traj(len(idx), restart, init_clone)

            new_score = self.comp_score(new_traj)

            # Find the actual length of each new trajectory (i.e. the first NaN)
            new_nt = np.argmax(np.isnan(new_traj), axis=1)
            new_nt[new_nt==0] = new_traj.shape[1]

            # Update the length of the resimulated trajectories
            Nt[idx] = restart + new_nt
            new_max = np.max(Nt[idx])

            # If the nex max length is greater than the current max length, we need to pad the arrays
            # We need to add new_max - max_Nt NaNs at the end of each trajectory to make sure the array traj is large enough
            if new_max > max_Nt:
                if self.dimension > 1:
                    traj = np.concatenate((traj, np.full((self.N,new_max-max_Nt,self.dimension),np.nan)), axis=1)
                else:
                    traj = np.concatenate((traj, np.full((self.N,new_max-max_Nt),np.nan)), axis=1)
                score = np.concatenate((score, np.full((self.N,new_max-max_Nt),np.nan)), axis=1)
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

        return dict({'probability':w*count_collapse/self.N,
                     'iterations':k,
                     'nb_transitions':count_collapse,
                     'trajectories':traj,
                     'scores':score})
