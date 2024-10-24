import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from compare_results import plot_committor_estimate, plot_PB

def walk_gradient_continuous(Z, dZdx, dZdy, start_x, start_y, x_values, y_values, ascent=True, step_size=0.1, max_steps=1000, tolerance=1e-2):
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
    
    return path_x, path_y

def get_gradient_path(x_values: np.array, y_values: np.array, Z: np.array, start_x: float, start_y: float, ascent=True, step_size=0.01, max_steps=500, tolerance=1e-2):
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
    path_x, path_y = walk_gradient_continuous(Z, dZdx, dZdy, start_x, start_y, x_values, y_values, ascent, step_size, max_steps, tolerance)

    return path_x, path_y

def plot_steepest_gradient(fig, ax, file_in, t_start, x_start):
    data = np.loadtxt(file_in, delimiter=';')
    t = data[:, 0]
    x = data[:, 1]
    prob = data[:, 2]
    prob[prob == 0] = 1e-18   
    time_grid = np.sort(np.unique(t))
    position_grid = np.sort(np.unique(x))
    prob_grid = prob.reshape((len(time_grid), len(position_grid)))
    path_x_plot, path_y_plot = get_gradient_path(time_grid, position_grid, prob_grid, t_start, x_start, ascent=True)
    ax.scatter(path_x_plot, path_y_plot, color='blue', s=10, label='Gradient Walk Path')




if __name__ == '__main__':
    # Load the data
    data = np.loadtxt('../temp/simulationAMS_runs5_grid1200.txt', delimiter=';')

    # Separate time, position, and probability
    time = data[:, 0]
    position = data[:, 1]
    prob = data[:, 2]
    
    # Avoid log(0)
    prob[prob == 0] = 1e-18   
    
    # Create grids for time and position
    time_grid = np.sort(np.unique(time))
    position_grid = np.sort(np.unique(position))
    
    # Reshape probability to match the grid
    prob_grid = prob.reshape((len(time_grid), len(position_grid)))

    # Calculate the steepest gradient path (use continuous starting point)
    time_start = 1.3  # Continuous value for time start
    x_start = -0.9    # Continuous value for position start

    path_x_plot, path_y_plot = get_gradient_path(time_grid, position_grid, prob_grid, time_start, x_start, ascent=True)

    # Plot the result
    plt.figure(figsize=(8, 8))
    plt.tricontourf(time, position, prob, levels=50, cmap='viridis')  # Contour plot of Z
    plt.plot(path_x_plot, path_y_plot, color='red', lw=2, label='Gradient Walk Path')
    plt.scatter(path_x_plot, path_y_plot, color='blue', s=10)  # Highlight the path points
    plt.title('Steepest Gradient Path (Descent)')
    plt.xlabel('Time')
    plt.ylabel('Position')
    c = plt.colorbar()
    c.set_label('Probability')
    plt.legend()
    plt.show()
