import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline



def walk_gradient_continuous(Z_interp, dZdx_interp, dZdy_interp, start_x, start_y, x_values, y_values, ascent=True, step_size=0.01, max_steps=10000, tolerance=1e-8):
    path_x = [start_x]
    path_y = [start_y]
    
    current_x = start_x
    current_y = start_y
    
    for step in range(max_steps):
        # Interpolate the gradient at the current position using the interpolators
        grad_x = dZdx_interp(current_x, current_y) 
        grad_y = dZdy_interp(current_x, current_y)

        # Compute the magnitude of the gradient
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Stop if the gradient magnitude is smaller than the tolerance
        
        if grad_magnitude < tolerance:
            print(f"Stopped at step {step}: gradient magnitude is too small ({grad_magnitude.item()}).")
            break
        

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
        path_x.append(current_x.item())
        path_y.append(current_y.item())

        if step == max_steps - 1:
            print(f"Stopped at step {step}: max steps reached.")

    path = np.array([path_x, path_y]).T
    return path

def get_gradient_path(x_values: np.array, y_values: np.array, Z: np.array, start_x: float, start_y: float, ascent=True):
    '''
    Compute the steepest gradient path on a given grid with interpolation.
    '''
    # Interpolators for Z and its gradients
    Z_interp = RectBivariateSpline(x_values, y_values, Z)
    dZdx_interp = RectBivariateSpline(x_values, y_values, np.gradient(Z, axis=0))  # Gradient along x
    dZdy_interp = RectBivariateSpline(x_values, y_values, np.gradient(Z, axis=1))  # Gradient along y

    # Walk the gradient and return the path
    path = walk_gradient_continuous(Z_interp, dZdx_interp, dZdy_interp, start_x, start_y, x_values, y_values, ascent)

    return path



#---------------------------------------------------------------

def plot_gradient_path(fig, ax, file_in, t_start, x_start):
    data = np.loadtxt(file_in, delimiter=';')
    time_grid = np.sort(np.unique(data[:, 0]))
    position_grid = np.sort(np.unique(data[:, 1]))
    prob = data[:, 2]
    prob[prob == 0] = 1e-22   
    #prob = np.log(prob)
    prob_grid = prob.reshape((len(time_grid), len(position_grid)))
    path = get_gradient_path(time_grid, position_grid, prob_grid, t_start, x_start, ascent=True)
    ax.scatter(t_start, x_start, color='red', label='Starting point', marker='x')
    ax.plot(path[:,0], path[:,1], color='red', label='Gradient path (ascent)')
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
    fig, ax = plot_gradient_path(fig, ax, file_committor, t_start=3.4, x_start=-0.35)
    fig, ax = plot_committor_gradient(fig, ax, file_committor)
    ax.legend()
    file_out = "../temp/reconstruct.png"
    fig.savefig(file_out)
    print('Saving to...', file_out)

    fig2, ax2 = plt.subplots(dpi=250)
    fig2, ax2 = plot_committor_gradient(fig2, ax2, file_committor)
    fig2, ax2 = plot_committor_estimate(fig2, ax2, file_committor, nb_runs=5, levels=levels, lognorm=False)
    fig2, ax2 = plot_PB(fig2, ax2, model)
    fig2.savefig("../temp/gradient.png")
    print('Saving to...', "../temp/gradient.png")




