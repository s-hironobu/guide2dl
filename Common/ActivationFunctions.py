#
# Activation and their Derivative functions
#
# Developed environment:
# Python                       3.11.5
# keras                        2.15.0
# pip                          24.0
# numpy                        1.26.4
# matplotlib                   3.9.0
# tensorflow                   2.15.1
# tensorflow-metal             1.1.0
# scikit-learn                 1.5.0
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
