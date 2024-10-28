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
        self.dx = self.x[1] - self.x[0]
        self.y = np.sort(np.unique(self.data[:, 1]))
        self.dy = self.y[1] - self.y[0]
        Z = np.where(self.data[:,2] == 0, 1e-22, self.data[:,2]) # Avoid log(0)
        self.Z = Z.reshape((len(self.y), len(self.x)))
        self.X, self.Y = np.meshgrid(self.x, self.y) #note: capital letters

    def set_gradients(self, method='nearest'):
        Z = ndimage.gaussian_filter(self.Z, sigma=0.8) #helps
        self.dZdx = ndimage.sobel(Z, axis=1) #this works
        self.dZdy = ndimage.sobel(Z, axis=0) #this works
        self.dZdx_interp = RegularGridInterpolator((self.y, self.x), self.dZdx*self.dx, bounds_error=False, fill_value=np.nan, method=method)
        self.dZdy_interp = RegularGridInterpolator((self.y, self.x), self.dZdy*self.dy, bounds_error=False, fill_value=np.nan, method=method)

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
                print(f'Out of bounds at step {i}')
                break
            path.append([pos[0,0], pos[0,1]]) 

            if i == steps-1:
                print(f'Ended regularly at step {i}')    
                
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
    file_committor_2 = "../temp/simulation_grid4860_noise_e2.txt" #less noise

    committor_1 = committor(file_committor_1)
    committor_2 = committor(file_committor_2)
    committor_1.set_gradients()
    committor_2.set_gradients()
    path_1 = committor_1.gradient_ascent(0.5, -1.0, lr=0.01)
    path_2 = committor_2.gradient_ascent(0.5, -0.999, lr=0.01)

    fig1, ax1 = plt.subplots(figsize=(7, 7), dpi=250)
    fig1, ax1 = committor_1.plot_contour(fig1, ax1)
    fig1, ax1 = committor_1.plot_gradient(fig1, ax1)
    fig1, ax1 = committor_1.plot_ascent_path(fig1, ax1, path_1)
    fig1, ax1 = plot_PB(fig1, ax1, model)
    ax1.set_title(r'More noise: $g=0.1$')
    ax1.legend()
    fig1.savefig("../temp/committor_noise_e1.png")
    print('Saved to ../temp/committor_noise_e1.png')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=250)
    fig2, ax2 = committor_2.plot_contour(fig2, ax2)
    fig2, ax2 = committor_2.plot_gradient(fig2, ax2)
    fig2, ax2 = committor_2.plot_ascent_path(fig2, ax2, path_2)
    fig2, ax2 = plot_PB(fig2, ax2, model)
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(-1.0, -0.99)
    ax2.set_title(r'Less noise: $g=0.01$')
    ax2.legend()
    fig2.savefig("../temp/committor_noise_e2.png")
    print('Saved to ../temp/committor_noise_e2.png')
    plt.close(fig2)
    




