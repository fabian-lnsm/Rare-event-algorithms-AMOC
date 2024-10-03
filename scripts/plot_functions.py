
import numpy as np
import matplotlib.pyplot as plt

class plot_DoubleWell():
    def __init__(self, t, mu):
        self.mu = mu
        self.t = t
    
    def V(self, x):
        return 0.25 * x**4 - 0.5 * x**2 - self.mu * x * self.t

    def plot(self):
        x = np.linspace(-2, 2, 100)
        fig,ax = plt.subplots()
        ax.plot(x, self.V(x), label='mu = '+str(self.mu)+', t = '+str(self.t))
        ax.set_xlabel('x')
        ax.set_ylabel('V(x)')
        ax.legend()
        fig.savefig(f'../results/figures/DoubleWell_potential_t_{self.t:.0f}'+'.png')

    
class plot_trajectories():
    def __init__(self, traj, params: dict, filename: str):
        self.traj = traj
        self.params = params
        self.filename = filename
    
    def plot(self):
        fig,ax = plt.subplots(dpi=200)
        fig.suptitle(str(self.params))
        for i in range(self.traj.shape[0]):
            ax.plot(self.traj[i,:,0], self.traj[i,:,1], label='Trajectory '+str(i+1))
        ax.set_xlabel('Time t')
        ax.set_ylabel('Position x')
        ax.grid()
        #ax.legend()
        fig.savefig(self.filename)

if __name__ == "__main__":


    t = 250
    mu = 0.001
    plot_DoubleWell(t, mu).plot()
