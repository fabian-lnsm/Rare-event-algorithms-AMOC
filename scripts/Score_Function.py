import numpy as np

class score_x:
    """
    Assumes that the equilibrium states are at x=1 and x=-1
    """

    def __init__(self):
        self.equilibrium = 1  # abs(x-coordinate) of the equilibrium states

    def score(self, traj : np.array):
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
