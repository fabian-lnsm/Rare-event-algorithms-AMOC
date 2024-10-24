import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline



def walk_gradient_continuous(Z, dZdx, dZdy, start_x, start_y, x_values, y_values, ascent=True, step_size=0.001, max_steps=2000, tolerance=1e-2):
    path_x = [start_x]
    path_y = [start_y]
    
    current_x = start_x
    current_y = start_y
    
    for step in range(max_steps):
        # Find the current position in the grid's coordinate system
        x_idx = (current_x - x_values[0]) / (x_values[-1] - x_values[0]) * (len(x_values) - 1)
        y_idx = (current_y - y_values[0]) / (y_values[-1] - y_values[0]) * (len(y_values) - 1)
        
        # Interpolate the gradient at the current position
        grad_x = map_coordinates(dZdx, [[y_idx], [x_idx]], order=1)[0]
        grad_y = map_coordinates(dZdy, [[y_idx], [x_idx]], order=1)[0]
        
        # Compute the magnitude of the gradient
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Stop if the gradient magnitude is smaller than the tolerance
        '''
        if grad_magnitude < tolerance:
            print(f"Stopped at step {step}: gradient magnitude is too small ({grad_magnitude:.2e}).")
            break
        '''
        
        # Determine the step direction (ascent or descent)
        if ascent:
            step_x = grad_x / grad_magnitude  # Move in the gradient direction (normalized)
            step_y = grad_y / grad_magnitude
        else:
            step_x = -grad_x / grad_magnitude  # Move in the opposite direction for descent
            step_y = -grad_y / grad_magnitude
        
        # Update the current position
        current_x += step_size * step_x
        current_y += step_size * step_y
        
        # Ensure we stay within the grid bounds
        if not (x_values[0] <= current_x <= x_values[-1] and y_values[0] <= current_y <= y_values[-1]):
            print(f"Stopped at step {step}: out of bounds.")
            break
        
        # Store the new position in the path
        path_x.append(current_x)
        path_y.append(current_y)

        if step==max_steps-1:
            print(f"Stopped at step {step}: max steps reached.")
    
    return path_x, path_y

def get_gradient_path(x_values: np.array, y_values: np.array, Z: np.array, start_x: float, start_y: float, ascent=True):
    '''
    Description
    -----------
    Compute the steepest gradient path on a given grid.


    Parameters
    ----------
    x_values : np.array of shape (N,)
        The x-coordinates of the grid.
    y_values : np.array of shape (M,)
        The y-coordinates of the grid.
    Z : np.array of shape (M * N)
        The grid values for which we calculate the gradient.
    start_x : float
        The starting x-coordinate for the path (can be continuous, not on the grid).
    start_y : float
        The starting y-coordinate for the path (can be continuous, not on the grid).
    ascent : bool
        If True, the path will follow the gradient ascent. Choose False for descent.
    step_size : float
        Step size for the gradient walk.
    max_steps : int
        Maximum number of steps to take.
    tolerance : float
        Stop when the gradient magnitude falls below this value.

    Returns
    -----------
    path_x : np.array of shape (L,)
        The x-coordinates of the path.
    path_y : np.array of shape (L,)
        The y-coordinates of the path.
    '''
    # Compute the gradients of Z with respect to x and y
    dZdx, dZdy = np.gradient(Z, x_values, y_values)  

    # Walk the gradient and return the path
    path_x, path_y = walk_gradient_continuous(Z, dZdx, dZdy, start_x, start_y, x_values, y_values, ascent)

    return path_x, path_y


#---------------------------------------------------------------

def plot_gradient_path(fig, ax, file_in, t_start, x_start):
    data = np.loadtxt(file_in, delimiter=';')
    time_grid = np.sort(np.unique(data[:, 0]))
    position_grid = np.sort(np.unique(data[:, 1]))
    prob = data[:, 2]
    prob[prob == 0] = 1e-22   
    prob_grid = prob.reshape((len(time_grid), len(position_grid)))
    path_x_plot, path_y_plot = get_gradient_path(time_grid, position_grid, prob_grid, t_start, x_start, ascent=True)
    ax.scatter(t_start, x_start, color='red', label='Starting point', marker='x')
    ax.plot(path_x_plot, path_y_plot, color='red', label='Gradient path (ascent)')
    return fig, ax


def plot_committor_estimate(fig, ax, file_in, nb_runs, levels, lognorm = True):
        data = np.loadtxt(file_in, delimiter=";")
        t_init = data[:, 0]
        x_init = data[:, 1]
        probabilities = data[:, 2]
        probabilities[probabilities == 0] = 1e-22   # Avoid log(0)
        resolution = [np.size(np.unique(t_init)), np.size(np.unique(x_init))]
        fig.suptitle(f'AMS: runs = {nb_runs}, resolution = {resolution}')
        norm = None
        if lognorm is True:
            norm = LogNorm()
        t=ax.tricontourf(
            t_init, x_init, probabilities,
            alpha=0.7, cmap="viridis",
            levels=levels,
            norm=norm
        )
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel(r"$x_{init}$")
        ax.set_xlim(np.min(t_init), np.max(t_init))
        ax.set_ylim(np.min(x_init), np.max(x_init))
        ax.grid(True)
        cbar = fig.colorbar(t)
        #cbar.set_label("Committor estimate")
        return fig, ax


def plot_PB(fig, ax, model):
    PB_traj = model.get_pullback(return_between_equil = True)
    ax.plot(
            PB_traj[:, 0],
            PB_traj[:, 1],
            label="PB_attractor",
            color="black", linewidth=2
            )
    return fig, ax

def plot_committor_gradient(fig, ax, file_in):
    data = np.loadtxt(file_in, delimiter=';')
    time_grid = np.sort(np.unique(data[:, 0]))
    position_grid = np.sort(np.unique(data[:, 1]))
    prob = data[:, 2]
    prob[prob == 0] = 1e-22   
    prob_grid = prob.reshape((len(time_grid), len(position_grid)))
    dZdx, dZdy = np.gradient(prob_grid, time_grid, position_grid)
    T, X = np.meshgrid(time_grid, position_grid)
    ax.quiver(T, X, dZdx, dZdy, color='red', label='Gradient')
    return fig, ax

#---------------------------------------------------------------
if __name__ == "__main__":
    from DoubleWell_Model import DoubleWell_1D
    model = DoubleWell_1D(mu = 0.03)
    file_committor = "../temp/simulationAMS_runs5_grid1200.txt"
    levels = np.linspace(0, 1, num=41)


    fig, ax = plt.subplots(dpi=250)
    fig, ax = plot_PB(fig, ax, model)
    fig, ax = plot_committor_estimate(fig, ax, file_committor, nb_runs=5, levels=levels, lognorm=False)
    fig, ax = plot_gradient_path(fig, ax, file_committor, t_start=2.3, x_start=-0.95)
    fig, ax = plot_committor_gradient(fig, ax, file_committor)
    ax.legend()
    file_out = "../temp/reconstruct.png"
    fig.savefig(file_out)
    print('Saving to...', file_out)

    fig2, ax2 = plt.subplots(dpi=250)
    fig2, ax2 = plot_committor_gradient(fig2, ax2, file_committor)
    fig2, ax2 = plot_committor_estimate(fig2, ax2, file_committor, nb_runs=5, levels=levels, lognorm=False)
    fig2.savefig("../temp/gradient.png")
    print('Saving to...', "../temp/gradient.png")




