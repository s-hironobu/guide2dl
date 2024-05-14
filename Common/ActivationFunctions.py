#
# Activation and their Derivative functions
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import numpy as np

#
# Sigmoid
#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


#
# Tanh
#
def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return (1 - np.tanh(x)**2)

#
# Linear
#
def linear(x):
    return x

def deriv_linear(x):
    return np.ones(x.shape)

#
# ReLU
#
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return np.where(x > 0, 1, 0)


#
# Softmax
#
# Assume that the loss function is the Cross-Entropy (CE).
#
class Softmax:

    def activate_func(self, x):
        exp = np.exp(x)
        self.y = exp / (np.sum(exp, axis=0) + 1e-8)
        return self.y

    # dL should be (- Label / output) because of CE.
    def deriv_activate_func(self, _, dL):
        return self.y * (1 + dL)
