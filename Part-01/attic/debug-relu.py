#
# Debugging for ReLU
#
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import autograd.numpy as np
from autograd import grad

units_size = 10

#
# Prepare inputs and outputs
#

x = np.arange(units_size) / 10 - units_size/ 20
x = x.reshape(units_size, 1)

Y = np.arange(units_size) / 10 - units_size/ 20
Y = Y.reshape(units_size, 1)

# ===================
# ReLU
# ===================

y = np.maximum(0, x)
Y = np.maximum(0, x) * 0.1

################################################
# Back Propagation
################################################

dL = (y - Y)

dh = np.where(x > 0, 1, 0) * dL

print("========== BP ==========")

print("dh =", dh)

################################################
# CHECK
################################################

print("========== Autograd ==========")

def relu(x):
    return np.maximum(0, x)


def loss(x, Y):
    return np.sum((relu(x) - Y) ** 2 / 2)

gradient_fun = grad(loss)

weights = gradient_fun(x, Y)
print("{} = {}".format("dh", weights))
print("\t=> OK") if np.allclose(weights, dh) else print("***** ERROR!!!")

