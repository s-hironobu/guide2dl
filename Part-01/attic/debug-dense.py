#
# Debugging for Dense
#
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import autograd.numpy as np
from autograd import grad

input_units = 2
hidden_units = 3

#
# Define functions
#
def tanh(Z):
    return np.tanh(Z)

def deriv_tanh(Z):
    return 1.0 - np.tanh(Z) ** 2

#
# Initialize Weights and Bias
#
W = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
b = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10

#
# Prepare inputs and outputs
#

x = np.arange(input_units) / 10
x = x.reshape(input_units, 1)

Y = np.arange(hidden_units) / 20
Y = Y.reshape(hidden_units, 1)

# ===================
# Forward Propagation
# ===================

# Dense
h = np.dot(W, x) + b
y = tanh(h)

################################################
# Back Propagation
################################################

dW = np.zeros_like(W)
db = np.zeros_like(b)

dL = (y - Y)

# Dense
dW = np.zeros_like(W)
db = np.zeros_like(b)
dx = np.zeros_like(x)

db = deriv_tanh(h) * dL
dW = np.dot(db, x.T)
dx = np.dot(W.T, db)

print("========== BP ==========")

print("dW =", dW)
print("db =", db)
print("dx =", dx)

################################################
# CHECK
################################################

"""
h = np.dot(W, x) + b
y = tanh(np.dot(W, x) + b)
"""

def dense(W, b, x):
    return tanh(np.dot(W, x) + b)

def loss(_W, y):
    return np.sum((forwardprop(_W) - Y) ** 2 / 2)

gradient_fun = grad(loss)

def eval_grad(_W, _dW, param):
    weights = gradient_fun(_W, Y)
    print("{} = {}".format(str(param), weights))
    print("\t=> OK") if np.allclose(weights, _dW) else print("***** ERROR!!!")

print("========== Autograd ==========")

###############
# W, b, x
###############

# W
def forwardprop(W):
    return dense(W, b, x)

eval_grad(W, dW, "dW")

# b
def forwardprop(b):
    return dense(W, b, x)

eval_grad(b, db, "db")

# x
def forwardprop(x):
    return dense(W, b, x)

eval_grad(x, dx, "dx")
