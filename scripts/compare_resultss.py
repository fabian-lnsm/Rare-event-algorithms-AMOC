import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator

class committor():
    def __init__(self, file_in):
        self.file_in = file_in
        self.data = np.loadtxt(file_in, delimiter=';')
        self.x = np.sort(np.unique(self.data[:, 0]))
        self.y = np.sort(np.unique(self.data[:, 1]))
        Z = np.where(self.data[:,2] == 0, 1e-22, self.data[:,2]) # Avoid log(0)
        self.Z = Z.reshape((len(self.y), len(self.x)))
        self.X, self.Y = np.meshgrid(self.x, self.y) #note: capital letters

    def set_gradients(self, method='nearest'):
        Z = ndimage.gaussian_filter(self.Z, sigma=0.8) #helps
        self.dZdx = ndimage.sobel(Z, axis=1) #this works
        self.dZdy = ndimage.sobel(Z, axis=0) #this works
        self.dZdx_interp = RegularGridInterpolator((self.y, self.x), self.dZdx, bounds_error=False, fill_value=np.nan, method=method)
        self.dZdy_interp = RegularGridInterpolator((self.y, self.x), self.dZdy, bounds_error=False, fill_value=np.nan, method=method)

    def gradient_ascent(self, x_start, y_start, steps=1000, lr=0.01):
        path=[]
        pos = np.array([[x_start, y_start]]) 
        path.append([pos[0,0], pos[0,1]]) 
        for i in range(steps):
            grad_x = self.dZdx_interp([pos[0,1], pos[0,0]]).item()
            grad_y = self.dZdy_interp([pos[0,1], pos[0,0]]).item()
            grad = np.array([grad_x, grad_y])
            grad = grad / np.linalg.norm(grad)
            pos+= lr * grad
            if pos[0,0] < np.min(self.x) or pos[0,0] > np.max(self.x) or pos[0,1] < np.min(self.y) or pos[0,1] > np.max(self.y):
                print(f'Out of bounds at step {i} with position {pos[0,0], pos[0,1]}')
                break
            path.append([pos[0,0], pos[0,1]]) 

            if i % 100 == 0:
                print(f'Step {i} with position {pos[0,0], pos[0,1]}')
            if i == steps-1:
                print(f'Ended regularly at step {i} with: {pos[0,0], pos[0,1]}')    
        path=np.array(path)
        return path

    def plot_ascent_path(self, fig, ax, path):
        ax.plot(path[:, 0], path[:, 1], color='blue', label='Gradient ascent', linewidth=2)
        return fig, ax

    def plot_gradient(self, fig, ax):
        ax.quiver(self.X, self.Y, self.dZdx, self.dZdy, color='red', label='Gradient field')
        return fig, ax
    
    def plot_contour(self, fig, ax, lognorm = False, levels=None):
        norm = None
        if lognorm is True:
            norm = LogNorm()
        if levels is None:
            levels = 100
        t=ax.contourf(
            self.X, self.Y, self.Z, levels=levels, alpha=0.8, cmap='viridis', norm=norm
            )
        ax.set_xlabel(r"$t_{init}$")
        ax.set_ylabel(r"$x_{init}$")
        ax.set_xlim(np.min(self.x), np.max(self.x))
        ax.set_ylim(np.min(self.y), np.max(self.y))
        ax.grid(True)
        cbar = fig.colorbar(t)
        #cbar.set_label("Committor estimate")
        return fig, ax
        

#---------------------------------------------------------------

def plot_PB(fig, ax, model):
    PB_traj = model.get_pullback(return_between_equil = True)
    ax.plot(
            PB_traj[:, 0],
            PB_traj[:, 1],
            label="PB attractor",
            color="black", linewidth=2
            )
    return fig, ax

#---------------------------------------------------------------

if __name__ == "__main__":
    from DoubleWell_Model import DoubleWell_1D
    model = DoubleWell_1D(mu = 0.03)

    file_committor_1 = "../temp/simulationAMS_runs5_grid1200_noise_e1.txt" #more noise
    file_committor_2 = "../temp/simulationAMS_runs5_grid1200_noise_e2.txt" #less noise

    committor_1 = committor(file_committor_1)
    committor_2 = committor(file_committor_2)
    committor_1.set_gradients()
    committor_2.set_gradients()
    path_1 = committor_1.gradient_ascent(0.5, -0.9)
    path_2 = committor_2.gradient_ascent(0.5, -0.9)

    fig1, ax1 = plt.subplots(nrows=2, figsize=(7, 14), dpi=250, sharex=True, sharey=True)
    fig1, ax1[0] = committor_1.plot_contour(fig1, ax1[0])
    fig1, ax1[0] = committor_1.plot_gradient(fig1, ax1[0])
    fig1, ax1[0] = committor_1.plot_ascent_path(fig1, ax1[0], path_1)
    fig1, ax1[0] = plot_PB(fig1, ax1[0], model)
    ax1[0].set_title('More noise')
    ax1[0].tick_params(labelbottom=False)
    ax1[0].legend()
    fig1, ax1[1] = committor_2.plot_contour(fig1, ax1[1], levels=np.linspace(0.5, 1, 100))
    fig1, ax1[1] = committor_2.plot_gradient(fig1, ax1[1])
    fig1, ax1[1] = committor_2.plot_ascent_path(fig1, ax1[1], path_2)
    fig1, ax1[1] = plot_PB(fig1, ax1[1], model)
    ax1[1].set_title('Less noise')
    fig1.tight_layout()
    fig1.savefig("../temp/committor_comparison.png")
    print('Saved to ../temp/committor_comparison.png')
    




