
import numpy as np

class x_coord():
    def __init__(self, model,init_state):
        '''
        model: instance of the model
        init_state: list of shape (2) with the initial state of the model
        '''
        self.model = model
        self.origin = np.abs(init_state[1]) #return the abs(x-coordinate) of the intitial state (=left equilibrium point)
       
    def process(self, traj):
        '''
        traj: np.array of shape (N,d) where N=No. of timesteps d=2 dimensions, with d[0]=time, d[1]=x
        '''
        x_value=traj[...,1]
        score=(x_value+self.origin)/(2*self.origin)
        return score


""" times=np.linspace(0,10,10)
positions=np.linspace(-1.0,1.0,10)
traj=np.vstack((times,positions)).T
print(traj.shape,traj)
scorefunc=x_coord(None,init_state=[0.0,-1.0])
score=scorefunc.process(traj)
print(score)  """
