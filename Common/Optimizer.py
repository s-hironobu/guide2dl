#
# Stochastic Optimizer(s)
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
# Gradients clipping.
#
def clip_gradient_norm(grads, max_norm=0.25):
    """
    g = (max_nrom/|g|)*g
    """
    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads

#
# Update weights.
#
def update_weights(layers, lr=0.01, optimizer=None, max_norm=None):

    grads = []
    params = []
    for layer in layers:
        grads.extend(layer.get_grads())
        params.extend(layer.get_params())

    # Clip gradient
    if max_norm is not None:
        grads = clip_gradient_norm(grads, max_norm)

    # Weights and Bias Update
    if optimizer == None:
        # gradient descent
        for i in range(len(grads)):
            params[i] -= lr * grads[i]
    else:
        # ADAM
        optimizer.update(params, grads)

#
# ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
# https://arxiv.org/pdf/1412.6980v8.pdf
#
class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr  # Learning rate
        self.beta1 = beta1  # m's attenuation rate
        self.beta2 = beta2  # v's attenuation rate
        self.iter = 0
        self.m = None  # Momentum
        self.v = None  # Adaptive learning rate

    def update(self, params, grads):

        if self.m is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = (self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter))
        for i in range(len(params)):
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * grads[i])
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
