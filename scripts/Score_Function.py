import numpy as np


class x_coord:
    """
    Assumes that the equilibrium states are at x=1 and x=-1
    """

    def __init__(self, model):
        """
        model: instance of the model
        """
        self.model = model
        self.equilibrium = 1  # abs(x-coordinate) of the equilibrium states

    def process(self, traj):
        """
        traj: np.array of shape (N,d) where N=No. of timesteps d=2 dimensions, with d[0]=time, d[1]=x
        """
        x_value = traj[..., 1]
        score = (x_value + self.equilibrium) / (2 * self.equilibrium)
        return score


""" times=np.linspace(0,10,10)
positions=np.linspace(-1.0,1.0,10)
traj=np.vstack((times,positions)).T
print(traj.shape,traj)
scorefunc=x_coord(None,init_state=[0.0,-1.0])
score=scorefunc.process(traj)
print(score)  """
