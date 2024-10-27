import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline



def walk_gradient_continuous(Z_interp, dZdx_interp, dZdy_interp, start_x, start_y, x_values, y_values, ascent=True, step_size=0.2, max_steps=5, tolerance=1e-8):
    path_x = [start_x]
    path_y = [start_y]
    
    current_x = start_x
    current_y = start_y
    
    for step in range(max_steps):
        #grad_x = dZdx_interp(current_x, current_y) 
        #grad_y = dZdy_interp(current_x, current_y)
        current_point = np.array([current_x, current_y]).T
        print(current_point)
        grad_x = dZdx_interp(current_point)
        grad_y = dZdy_interp(current_point)
        print(grad_x, grad_y)


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

def get_gradient_path(x_values, y_values, Z, start_x, start_y, bivariate=False, ascent=True):
    '''
    Compute the steepest gradient path on a given grid with interpolation.

    Shapes
    ------
    x_values: (N,) array
    y_values: (M,) array
    Z: (N, M) array
    '''

    if bivariate == True:
        Z_interp = RectBivariateSpline(x_values, y_values, Z)
        dZdx_interp = RectBivariateSpline(x_values, y_values, np.gradient(Z, axis=0)) 
        dZdy_interp = RectBivariateSpline(x_values, y_values, np.gradient(Z, axis=1))
    else:
        Z_interp = RegularGridInterpolator((x_values, y_values), Z)
        dZdx_interp = RegularGridInterpolator((x_values, y_values), np.gradient(Z, axis=0), method='nearest')
        dZdy_interp = RegularGridInterpolator((x_values, y_values), np.gradient(Z, axis=1), method='cubic')

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
    x_values = np.sort(np.unique(data[:, 0]))
    y_values = np.sort(np.unique(data[:, 1]))
    Z = data[:, 2]
    Z[Z == 0] = 1e-22   
    Z_grid = Z.reshape((len(x_values), len(y_values)))
    dZdx, dZdy = np.gradient(Z_grid, x_values, y_values)
    X, Y = np.meshgrid(x_values, y_values)
    ax.quiver(X, Y, dZdx, dZdy, color='red', label='Gradient')
    return fig, ax

def plot_committor_gradient_interpolate(fig, ax, file_in):
    data = np.loadtxt(file_in, delimiter=';')
    x_values = np.sort(np.unique(data[:, 0]))
    y_values = np.sort(np.unique(data[:, 1]))
    Z = data[:, 2]
    Z[Z == 0] = 1e-22   
    Z = Z.reshape((len(x_values), len(y_values)))
    dZdx_interp = RegularGridInterpolator((x_values, y_values), np.gradient(Z, axis=0), method='nearest')
    dZdy_interp = RegularGridInterpolator((x_values, y_values), np.gradient(Z, axis=1), method='nearest')
    x_values = np.linspace(x_values[0], x_values[-1], 100)
    y_values = np.linspace(y_values[0], y_values[-1], 100)
    X, Y = np.meshgrid(x_values, y_values)
    input_points = np.array([X, Y]).T
    dZdx = dZdx_interp(input_points)
    dZdy = dZdy_interp(input_points)
    ax.quiver(X, Y, dZdx, dZdy, color='red', label='Gradient')
    return fig, ax

#---------------------------------------------------------------
if __name__ == "__main__":
    from DoubleWell_Model import DoubleWell_1D
    model = DoubleWell_1D(mu = 0.03)
    file_committor = "../temp/simulationAMS_runs5_grid1200.txt"
    levels = np.linspace(0, 1, num=41)

    '''
    fig, ax = plt.subplots(dpi=250)
    fig, ax = plot_PB(fig, ax, model)
    fig, ax = plot_committor_estimate(fig, ax, file_committor, nb_runs=5, levels=levels, lognorm=False)
    fig, ax = plot_gradient_path(fig, ax, file_committor, t_start=3.4, x_start=-0.35)
    fig, ax = plot_committor_gradient_interpolate(fig, ax, file_committor)
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

    fig3, ax3 = plt.subplots(dpi=250)
    fig3, ax3 = plot_PB(fig3, ax3, model)
    fig3, ax3 = plot_committor_estimate(fig3, ax3, file_committor, nb_runs=5, levels=levels, lognorm=False)
    fig3, ax3 = plot_committor_gradient_interpolate(fig3, ax3, file_committor)
    ax3.legend()
    fig3.savefig("../temp/gradient_interpolate.png")
    print('Saving to...', "../temp/gradient_interpolate.png")
    '''
    def gradient_ascent(dZdx, dZdy, dx, dy, x_start, y_start, steps=1000, lr=0.1):
        path=[]
        pos = np.array([[x_start, y_start]]) 
        path.append([pos[0,0], pos[0,1]]) 
        for _ in range(steps):
            grad_x = dZdx([pos[0,1], pos[0,0]]).item()
            grad_y = dZdy([pos[0,1], pos[0,0]]).item()
            grad = np.array([grad_x, grad_y])
            grad = grad / np.linalg.norm(grad)
            print(pos, grad)
            pos+= lr * grad
            path.append([pos[0,0], pos[0,1]]) 
        path=np.array(path)
        return path

    
    fig4, axs4 = plt.subplots(nrows=2,figsize=(7, 14), dpi=250)
    data = np.loadtxt(file_committor, delimiter=';')
    x_axis = np.sort(np.unique(data[:, 0]))
    dx = x_axis[1] - x_axis[0]
    y_axis = np.sort(np.unique(data[:, 1]))
    dy = y_axis[1] - y_axis[0]
    Z = np.where(data[:,2] == 0, 1e-22, data[:,2])
    Z = Z.reshape((len(y_axis), len(x_axis)))
    Z = ndimage.gaussian_filter(Z, sigma=0.8) #helps
    X, Y = np.meshgrid(x_axis, y_axis)
    axs4[0].contourf(X, Y, Z, levels=levels, alpha=0.8)
    dZdx = ndimage.sobel(Z, axis=1) #this works
    dZdy = ndimage.sobel(Z, axis=0) #this works
    print(np.mean(dZdx), np.mean(dZdy), np.mean(Z), np.var(dZdx), np.var(dZdy), np.var(Z))
    axs4[0].quiver(X, Y, dZdx, dZdy, color='red', label='Gradient')

    dZdx_interp = RegularGridInterpolator((y_axis, x_axis), dZdx, method='nearest',bounds_error=False, fill_value=np.nan)
    dZdy_interp = RegularGridInterpolator((y_axis, x_axis), dZdy, method='nearest', bounds_error=False, fill_value=np.nan)
    path = gradient_ascent(dZdx_interp, dZdy_interp, dx, dy, x_start=0.2, y_start=-0.8)
    axs4[1].contourf(X, Y, Z, levels=100, alpha=0.8)
    axs4[1].scatter(path[:, 0], path[:, 1], color='lime', label='Gradient path')
    axs4[1].quiver(X, Y, dZdx, dZdy, color='red', label='Gradient', alpha=0.5)
    fig4.savefig("../temp/interpolation.png")
    print('Saving to...', "../temp/interpolation.png")


    
    
        

    
    fig5, axs5 = plt.subplots(nrows=2, figsize=(7, 14))
    x = np.linspace(0, 5, 30)
    dx = x[1] - x[0]
    y = np.linspace(0, 5, 40)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y**2 - 0.2*(X-2+Y)+0.05*Y**3)
    Z += 0.4 * np.random.normal(size=Z.shape)
    Z = ndimage.gaussian_filter(Z, sigma=0.8)
    dZdx = np.gradient(Z, dx, axis=1)
    dZdy = np.gradient(Z, dy, axis=0)
    print(np.mean(dZdx), np.mean(dZdy), np.mean(Z), np.var(dZdx), np.var(dZdy), np.var(Z))
    dZdx_interp = RegularGridInterpolator((y, x), dZdx, bounds_error=False, fill_value=np.nan)
    dZdy_interp = RegularGridInterpolator((y, x), dZdy, bounds_error=False, fill_value=np.nan)
    path = gradient_ascent(dZdx_interp, dZdy_interp, dx, dy, x_start=0.2, y_start=0.2, lr = 0.1)
    dZdx_true = 2*X
    dZdy_true = 2*Y
    axs5[0].contourf(X, Y, Z, levels=100)
    c=plt.colorbar(axs5[0].contourf(X, Y, Z, levels=50))
    axs5[0].plot(path[:, 0], path[:, 1], color='lime', label='Gradient path')
    axs5[0].quiver(X, Y, dZdx, dZdy, color='red', label='Gradient')
    #axs5[1].tricontourf(X.flatten(), Y.flatten(), Z.flatten(), levels=20)
    #axs5[1].quiver(X, Y, dZdx_true, dZdy_true, color='blue', label='True gradient')
    fig5.savefig("../temp/gradient_test.png")
    print('Saving to...', "../temp/gradient_test.png")
    
    




