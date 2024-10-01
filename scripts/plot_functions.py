

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

    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    t = 250
    mu = 0.001
    plot_DoubleWell(t, mu).plot()
