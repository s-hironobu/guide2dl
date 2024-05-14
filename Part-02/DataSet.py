#
# Data set generator.
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3

import numpy as np

# Create a sine wave.
def create_wave(n_data=100, noise=0.05):
    _x_range = np.linspace(0, 2 * 2 * np.pi, n_data)
    return np.sin(_x_range) + noise * np.random.randn(len(_x_range))

# Generate a data set.
def dataset(wave, n_sequence=25, return_sequences=False):

    n_sample = len(wave) - n_sequence

    x = np.zeros((n_sample, n_sequence, 1, 1))
    if return_sequences == True:
        y = np.zeros((n_sample, n_sequence, 1))
    else:
        y = np.zeros((n_sample, 1, 1))

    for i in range(n_sample):
        x[i, :, 0, 0] = wave[i : i + n_sequence]
        if return_sequences == True:
            y[i, :, 0] = wave[i + 1 : i + n_sequence + 1]
        else:
            y[i, 0] = wave[i + n_sequence]

    return x, y
