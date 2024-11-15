import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree



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
        score = np.clip(score, 0, 1)
        return score

class score_PB:
    '''
    Pullback attractor as a score function
    '''
    def __init__(self, model, decay_length):
        self.decay_length = decay_length
        self.model = model
        self.PB_trajectory = model.get_pullback(return_between_equil=True)
        self.score_function = self.score_function_maker(self.PB_trajectory, self.decay_length)


    def curvilinear_coordinate(self, reference_traj):
        '''
        Find the curvilinear coordinate of a trajectory

        Parameters
        ----------
        reference_traj : np.array of shape (timesteps, 2)

        Returns
        -------
        normalised_curvilinear_coordinate : np.array of shape (timesteps)

        '''
        
        nb_points = np.shape(reference_traj)[0]
        
        ds_2 = np.zeros(nb_points-1)
        
        for i in range(1, reference_traj.ndim):
            dxi = reference_traj[1:,i]-reference_traj[:-1,i]
            ds_2 = ds_2 + dxi**2
            
        curvilinear_coordinate = np.zeros(nb_points)
        curvilinear_coordinate[1:] = np.cumsum(np.sqrt(ds_2))
        normalised_curvilinear_coordinate = curvilinear_coordinate/curvilinear_coordinate[-1]
        
        return normalised_curvilinear_coordinate


    def score_function_maker(self, reference_traj, decay_length):
        '''
        Create a score function based on the pullback attractor
        
        Parameters
        ----------
        reference_traj : np.array of shape (timesteps, 2)
            The pullback attractor
        decay_length : float
            The decay length of the score function

        Returns
        -------
        score_function : function
            The score function
        '''
        normalised_curvilinear_coordinate = self.curvilinear_coordinate(reference_traj)
        kdtree = KDTree(reference_traj)


        def score(traj : np.array):
            '''
            Compute the score function for a given trajectory

            Parameters
            ----------
            traj : np.array of shape (..., 2)
                The trajectory for which to compute the score

            Returns
            -------
            scores: np.array of shape (...)
                The score for each trajectory point      
            '''

            original_shape = traj.shape[:-1]
            traj_reshape = traj.reshape(-1, 2) # reshape s.t. each entry is one point

            scores = np.full(traj_reshape.shape[0], np.nan)
            valid_mask = ~np.isnan(traj_reshape).any(axis=1)
            if valid_mask.any():
                closest_distances, closest_point_indices = kdtree.query(traj_reshape[valid_mask])
                s = normalised_curvilinear_coordinate[closest_point_indices] 
                scores[valid_mask] = s * np.exp(-(closest_distances / decay_length) ** 2)

            return scores.reshape(original_shape)
        
        return score
    
    def get_score(self, traj : np.array):
        return self.score_function(traj)

if __name__=='__main__':

    from DoubleWell_Model import DoubleWell_1D
    model = DoubleWell_1D(mu = 0.03)
    score = score_PB(model, decay_length = 0.2)


    traj = np.ones((5,10,2))
    traj[2:4,5:,:] = np.nan

    import time
    start = time.perf_counter()
    scores = score.get_score(traj)
    print(f'Elapsed time: {time.perf_counter()-start} for {np.prod(traj.shape[:-1])} points')
    print(scores)

