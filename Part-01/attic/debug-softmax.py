#
# Debugging for Softmax
#
#
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

import autograd.numpy as np
from autograd import grad

units_size = 10

#
# Prepare inputs and outputs
#

x = np.arange(units_size)/units_size - 0.5
x = x.reshape(units_size, 1)

Y = np.zeros(units_size)
Y[0] = 1.0
Y = Y.reshape(units_size, 1)

# ===================
# softmax
# ===================

exp = np.exp(x)
y = exp / (np.sum(exp, axis=0) + 1e-8)


################################################
# Back Propagation
################################################

dL = - Y / (y + 1e-8)

#dh = np.zeros(units_size)
dh = y * (1 + dL)

# Convert row vector into column vector.
dh = dh.reshape(units_size, 1)

print("========== BP ==========")

print("dh =", dh)


################################################
# CHECK
################################################

print("========== Autograd ==========")

def softmax(x):
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=0) + 1e-8)

def loss(x, Y):
    return -np.sum(Y * np.log(softmax(x) + 1e-8))

gradient_fun = grad(loss)

weights = gradient_fun(x, Y)
print("{} = {}".format("dh", weights))
print("\t=> OK") if np.allclose(weights, dh) else print("***** ERROR!!!")

