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
        score = np.clip(score, a_min=None, a_max=1)
        return score

class ScoreFunction_helper:
    '''
    Helper class for the score function. Needed for multiprocessing
    '''
    def __init__(self, reference_traj, normalised_curvilinear_coordinate, decay_length):
        self.kdtree = KDTree(reference_traj)
        self.normalised_curvilinear_coordinate = normalised_curvilinear_coordinate
        self.decay_length = decay_length

    def __call__(self, traj: np.array):
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
        traj_reshape = traj.reshape(-1, 2)

        scores = np.full(traj_reshape.shape[0], np.nan)
        valid_mask = ~np.isnan(traj_reshape).any(axis=1)
        if valid_mask.any():
            closest_distances, closest_point_indices = self.kdtree.query(traj_reshape[valid_mask])
            s = self.normalised_curvilinear_coordinate[closest_point_indices]
            scores[valid_mask] = s * np.exp(-(closest_distances / self.decay_length) ** 2)

        scores = np.clip(scores, a_min=None, a_max=1)
        scores = scores.reshape(original_shape)
        return scores

class score_PB:
    '''
    Pullback attractor as a score function
    '''
    def __init__(self, model, decay_length=0.2):
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
        ds_2 = np.zeros(nb_points - 1)
        for i in range(1, reference_traj.ndim):
            dxi = reference_traj[1:, i] - reference_traj[:-1, i]
            ds_2 = ds_2 + dxi ** 2

        curvilinear_coordinate = np.zeros(nb_points)
        curvilinear_coordinate[1:] = np.cumsum(np.sqrt(ds_2))
        normalised_curvilinear_coordinate = curvilinear_coordinate / curvilinear_coordinate[-1]

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
        return ScoreFunction_helper(reference_traj, normalised_curvilinear_coordinate, decay_length)

    def get_score(self, traj):
        return self.score_function(traj)

 
        
if __name__=='__main__':

    from DoubleWell_Model import DoubleWell_1D
    model = DoubleWell_1D(mu = 0.03)
    score_PB = score_PB(model, decay_length = 0.2)
    score_x = score_x()



    # Test the score function
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)
    positions = np.linspace(-1.5, 1.5, 100)
    times = np.linspace(0, 10, 100)
    positions, times = np.meshgrid(positions, times)
    states = np.stack([times, positions], axis=-1)
    scores = score_x.get_score(states)
    ax.contourf(times, positions, scores, cmap='viridis', levels=50)
    ax.set_title(r'Continuous $phi_x$')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Score')
    fig.savefig('../temp/score_x.png')


