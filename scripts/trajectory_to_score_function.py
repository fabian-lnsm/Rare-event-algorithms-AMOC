import numpy as np
from DoubleWell_Model import DoubleWell_1D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def curvilinear_coordinate(trajectory):
    
    nb_points = np.shape(trajectory)[0]
    
    ds_2 = np.zeros(nb_points-1)
    
    for i in range(1, trajectory.ndim):
        dxi = trajectory[1:,i]-trajectory[:-1,i]
        ds_2 = ds_2 + dxi**2
        
    curvilinear_coordinate = np.zeros(nb_points)
    curvilinear_coordinate[1:] = np.cumsum(np.sqrt(ds_2))
    normalised_curvilinear_coordinate = curvilinear_coordinate/curvilinear_coordinate[-1]
    
    return normalised_curvilinear_coordinate


def score_function_maker(trajectory, decay):
    normalised_curvilinear_coordinate = curvilinear_coordinate(trajectory)
    
    def score_function(v):
        distances = np.linalg.norm(trajectory-v, axis = 1)
        closest_point_index = np.argmin(distances)
    
        d = distances[closest_point_index] #distance between vector and trajectory,, normalised
        s = normalised_curvilinear_coordinate[closest_point_index]    #curvilinear coordinate corresponding to closest point
        
        score = s*np.exp(-(d/decay)**2)
        return score
    return score_function

def plot_PB_score_function(fig,ax, PB_trajectory, mu, score, levels):
    plt.plot(
        PB_trajectory[:,0],PB_trajectory[:,1],label=f'PB: Trajectory with mu = {mu}',
        color='black',linewidth=2,
        )
    t_map = np.linspace(np.min(PB_trajectory[:,0]),np.max(PB_trajectory[:,0]),200)
    x_map = np.linspace(np.min(PB_trajectory[:,1]),np.max(PB_trajectory[:,1]),200)
    X_map, T_map = np.meshgrid(x_map, t_map)
    Z = np.zeros_like(X_map)
    for i in range(len(t_map)):
        for j in range(len(x_map)):
            Z[i,j] = score([t_map[i],x_map[j]])
    Z[Z==0] += 1e-20
    cont = ax.contourf(
        T_map, X_map, Z,
        levels, cmap='viridis', alpha = 0.7,
        norm=LogNorm() 
        )
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    ax.grid()
    cbar = fig.colorbar(cont)
    cbar.set_label("Score function")
    ax.legend()
    return fig, ax



if __name__=='__main__':

    filepath = '../temp/trajectory_to_score_function.png'
    model = DoubleWell_1D()
    mu = 0.03
    decay_scorefct = 0.2
    PB_trajectory = model.get_pullback(mu, return_between_equil=True)
    score=score_function_maker(PB_trajectory,decay_scorefct)

    fig, ax = plt.subplots(dpi=200)
    levels = np.logspace(-6, 0, 31)
    fig, ax = plot_PB_score_function(fig, ax, PB_trajectory, mu, score, levels)
    fig.suptitle(r"$\Phi_{PB}$ with decay length = "+str(decay_scorefct))
    fig.savefig(filepath)
    print(f"Figure saved to {filepath}")


