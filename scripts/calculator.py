import numpy as np

def significance (mean1, mean2, std1, std2):
    mean_diff = np.abs(mean1 - mean2)
    std_diff = np.sqrt(std1**2 + std2**2)
    score = mean_diff/std_diff
    return score

def relative_error(mean, std):
    rel_error = std/mean
    return rel_error

means_x = np.array([6.9, 1.1, 6.5, 9.7])
std_x = np.array([5.6, 0.3, 0.5, 0.4])

means_PB = np.array([2.8, 1.6, 3.4, 6.6])
std_PB = np.array([2.7, 1.0, 1.0, 0.8])

means_MC = np.array([7.0, 1.5, 6.7, 9.5])
std_MC = np.array([1.2, 0.3, 0.7, 0.3])

print('Significance x vs MC: ',significance(means_x, means_MC, std_x, std_MC))
print('Relative error x: ',relative_error(means_x, std_x))

print('Significance PB vs MC: ',significance(means_PB, means_MC, std_PB, std_MC))
print('Relative error PB: ',relative_error(means_PB, std_PB))
