import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.colors import LogNorm



class score_x:
    """
    Assumes that the equilibrium states are at x=1 and x=-1
    """

    def __init__(self):
        self.equilibrium = 1  # abs(x-coordinate) of the equilibrium states

    def __str__(self):
        return f'Class: Simple x-score'
    
    def __repr__(self):
        return f'Class: Simple x-score'

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
    def __init__(self, model, decay_length):
        self.decay_length = decay_length
        self.model = model
        self.PB_trajectory = model.get_pullback(return_between_equil=True)
        self.score_function = self.score_function_maker(self.PB_trajectory, self.decay_length)

    def __str__(self):
        return f'Class: PB-score with decay length = {self.decay_length}'
    
    def __repr__(self):
        return f'Class: PB-score with decay length = {self.decay_length}'

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
    model.set_roots()

    

    def plot_PB(fig, ax, model):
        PB_traj = model.get_pullback(return_between_equil = True)
        ax.plot(
                PB_traj[:, 0],
                PB_traj[:, 1],
                label="PB attractor",
                color="black", linewidth=2
                )
        return fig, ax

    def plot_PB_score(decay_length):
        scorefct_PB = score_PB(model, decay_length = decay_length)
        t = np.linspace(0, 30, 300)
        x = np.linspace(-1.5, 1.6, 300)
        T, X = np.meshgrid(t, x)
        traj = np.stack((T, X), axis=-1)
        scores = scorefct_PB.get_score(traj)
        scores = np.where(scores == 0, 1e-22, scores)
        fig, ax = plt.subplots(dpi=250)
        contour=ax.contourf(
            T, X, scores, cmap='viridis', levels=np.linspace(0,1,51), alpha=0.85
            )
        cbar=fig.colorbar(contour)
        ax.set_xlabel('Time t')
        ax.set_ylabel('Position x')
        ax.set_title(f'PB score: Decay length = {scorefct_PB.decay_length:.2f}')
        fig, ax = plot_PB(fig, ax, model)
        ax.set_xlim(0, 22)
        ax.set_ylim(-1.1, 1.4)
        init_times = np.array([2.0, 4.0, 7.0, 10.0])
        init_positions = np.vectorize(model.on_dict.get)(init_times)
        init_states = np.stack([init_times, init_positions], axis=1)
        ax.scatter(init_states[:, 0], init_states[:, 1], color='black', label='Initial states', s=30, zorder=10)
        fig.savefig(f'../temp/PB_score_{decay_length:.2f}.png')


    plot_PB_score(1.5)

        


