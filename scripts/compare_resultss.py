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
        self.resolution = [len(self.x), len(self.y)]
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

    def plot_gradient(self, fig, ax, stepx=1, stepy=1 ):
        #ax.quiver(self.X, self.Y, self.dZdx, self.dZdy, color='red', label='Gradient field')
        ax.quiver(self.X[::stepy, ::stepx], self.Y[::stepy, ::stepx], self.dZdx[::stepy, ::stepx], self.dZdy[::stepy, ::stepx], color='red', label='Gradient field')
        return fig, ax
    
    def plot_contour(self, fig, ax, lognorm = False, levels=None):
        norm = None
        if lognorm is True:
            norm = LogNorm()
        if levels is None:
            levels = 100
        t=ax.contourf(
            self.X, self.Y, self.Z, levels=levels, alpha=1.0, cmap='viridis', norm=norm
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

    file_committor = "../temp/simulation_grid6040_noise_5e2.txt"
    comm = committor(file_committor)
    comm.set_gradients(method='nearest')
    #path = comm.gradient_ascent(1.5, -0.985, lr=0.01)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=250)
    fig, ax = comm.plot_contour(fig, ax, levels=40)
    #fig, ax = comm.plot_gradient(fig, ax, stepx=2, stepy=5)
    #fig, ax = comm.plot_ascent_path(fig, ax, path)
    fig, ax = plot_PB(fig, ax, model)
    ax.set_title(r"Medium noise: $g=0.05$; "+f' Res: {comm.resolution}')
    ax.legend()
    fig.savefig("../temp/noise_5e2.png")
    print('Figure saved')




