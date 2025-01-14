#
# GRU Model Validation with Autograd
#
#
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

import autograd.numpy as np
from autograd import grad

return_sequences = False

input_units = 1
hidden_units = 3
output_units = 1

#
# Define functions
#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(Z):
    return np.tanh(Z)

def deriv_tanh(Z):
    return 1.0 - np.tanh(Z) ** 2

def linear(Z):
    return Z

def deriv_linear(Z):
    return np.ones(Z.shape)


#
# Initialize Weights and Bias
#
Wz = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
Wr = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
W = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10

Uz = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10
Ur = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10
U = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10

bz = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
br = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
b = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10

V = np.arange(1, hidden_units * output_units + 1).reshape((output_units, hidden_units)) / 10
c = np.arange(1, output_units * 1 + 1).reshape((output_units, 1)) / 10

#
# Prepare inputs and outputs
#
n_sequence = 2

x = np.arange(n_sequence * input_units) + 10
if input_units != 1:
    x = x.reshape(n_sequence, input_units, 1)

if return_sequences == True:
    y = np.arange(n_sequence * output_units) * 10
    y = y.reshape(n_sequence, output_units, 1)
else:
    y = np.arange(output_units) * 10
    y = y.reshape(output_units, 1)

# ===================
# Forward Propagation
# ===================

# GRU
z = np.zeros([n_sequence, hidden_units, 1])
z_p = np.zeros([n_sequence, hidden_units, 1])
r = np.zeros([n_sequence, hidden_units, 1])
r_p = np.zeros([n_sequence, hidden_units, 1])
h_h = np.zeros([n_sequence, hidden_units, 1])
h_h_p = np.zeros([n_sequence, hidden_units, 1])
h = np.zeros([n_sequence, hidden_units, 1])

for t in range(n_sequence):
    if t == 0:
        z_p[t] = np.dot(Wz, x[t]) + bz
        z[t] = sigmoid(z_p[t])
        r_p[t] = np.dot(Wr, x[t]) + br
        r[t] = sigmoid(r_p[t])
        h_h_p[t] = np.dot(W, x[t]) + b
        h_h[t] = tanh(h_h_p[t])
        h[t] = z[t] * h_h[t]
    else:
        z_p[t] = (np.dot(Wz, x[t]) + np.dot(Uz, h[t - 1]) + bz)
        z[t] = sigmoid(z_p[t])
        r_p[t] = (np.dot(Wr, x[t]) + np.dot(Ur, h[t - 1]) + br)
        r[t] = sigmoid(r_p[t])
        h_h_p[t] = (np.dot(W, x[t]) + np.dot(U, r[t] * h[t - 1]) + b)
        h_h[t] = tanh(h_h_p[t])
        h[t] = (1 - z[t]) * h[t - 1] + z[t] * h_h[t]

# Dense
if return_sequences == True:
    y_h = np.zeros([n_sequence, output_units, 1])
    for s in range(n_sequence):
        y_h [s]= np.dot(V, h[s]) + c
    _y = linear(y_h)
else:
    y_h = np.dot(V, h[-1]) + c
    _y = linear(y_h)

################################################
# BPTT
################################################

dWz = np.zeros_like(Wz)
dWr = np.zeros_like(Wr)
dW = np.zeros_like(W)
dUz = np.zeros_like(Uz)
dUr = np.zeros_like(Ur)
dU = np.zeros_like(U)
dbz = np.zeros_like(bz)
dbr = np.zeros_like(br)
db = np.zeros_like(b)
dV = np.zeros_like(V)
dc = np.zeros_like(c)
dh = np.zeros([n_sequence, hidden_units, 1])

dL = (_y - y)

# Dense
if return_sequences == True:
    dc = deriv_linear(y_h) * dL
    grads = np.zeros([n_sequence, hidden_units, 1])
    for s in range(n_sequence):
        grads[s] = np.dot(V.T, dc[s])
else:
    dc = deriv_linear(y_h[-1]) * dL
    grads = np.dot(V.T, dc)

# GRU

for t in reversed(range(n_sequence)):

    if t == n_sequence - 1:
        if return_sequences == True:
            dh[t] = grads[t]
        else:
            dh[t] = grads
    else:
        d1 = np.dot(Uz.T, dh[t + 1] * (h_h[t + 1] - h[t]) * deriv_sigmoid(z_p[t + 1]))
        d2 = np.dot(r[t + 1] * U.T, dh[t + 1] * z[t + 1] * deriv_tanh(h_h_p[t + 1]))
        d3 = np.dot(Ur.T, np.dot(h[t] * U.T, dh[t + 1] * z[t + 1] * deriv_tanh(h_h_p[t + 1])) * deriv_sigmoid(r_p[t + 1]))
        d4 = dh[t + 1] * (1 - z[t + 1])

        dh[t] = d1 + d2 + d3 + d4
        if return_sequences == True:
            dh[t] += grads[t]

for t in range(n_sequence):

    if t > 0:
        _dbz = (dh[t] * deriv_sigmoid(z_p[t]) * (h_h[t] - h[t - 1]))
    else:
        _dbz = dh[t] * deriv_sigmoid(z_p[t]) * h_h[t]

    dbz += _dbz
    dWz += np.dot(_dbz, x[t].T)
    if t > 0:
        dUz += np.dot(_dbz, h[t - 1].T)

    if t > 0:
        _dbr = np.dot(h[t - 1] * U.T, dh[t] * z[t] * deriv_tanh(h_h_p[t])) * deriv_sigmoid(r_p[t])
        dbr += _dbr
        dWr += np.dot(_dbr, x[t].T)
        dUr += np.dot(_dbr, h[t - 1].T)

    _db = dh[t] * z[t] * deriv_tanh(h_h_p[t])
    db += _db
    dW += np.dot(_db, x[t].T)
    if t > 0:
        dU += np.dot(_db, (r[t] * h[t - 1]).T)


print("========== BPTT ==========")

print("dWz =", dWz)
print("dWr =", dWr)
print("dW =", dW)
print("dUz =", dUz)
print("dUr =", dUr)
print("dU =", dU)
print("dbz =", dbz)
print("dbr =", dbr)
print("db =", db)

################################################
# Validate
################################################

"""
# GRU
z_p[0] = np.dot(Wz, x[0]) + bz
z[0] = sigmoid(z_p[0])
r_p[0] = np.dot(Wr, x[0]) + br
r[0] = sigmoid(r_p[0])
h_h_p[0] = np.dot(W, x[0]) + b
h_h[0] = tanh(h_h_p[0])
h[0] = z[0] * h_h[0]

z_p[1] = (np.dot(Wz, x[1]) + np.dot(Uz, h[0]) + bz)
z[1] = sigmoid(z_p[1])
r_p[1] = (np.dot(Wr, x[1]) + np.dot(Ur, h[0]) + br)
r[1] = sigmoid(r_p[1])
h_h_p[1] = (np.dot(W, x[1]) + np.dot(U, r[1] * h[0]) + b)
h_h[1] = tanh(h_h_p[1])
h[1] = (1 - z[1]) * h[0] + z[1] * h_h[1]

# Dense
# y[0]
y_h = np.dot(V, h[0]) + c
_y = linear(y_h)

# y[1]
y_h = np.dot(V, h[1]) + c
_y = linear(y_h)
"""

def z_p_0(Wz, bz):
    # z_p[0] = np.dot(Wz, x[0]) + bz
    return np.dot(Wz, x[0]) + bz

def z_0(Wz, bz):
    # z[0] = sigmoid(z_p[0])
    return sigmoid(z_p_0(Wz, bz))

def r_p_0(Wr, br):
    # r_p[0] = np.dot(Wr, x[0]) + br
    return  np.dot(Wr, x[0]) + br

def r_0(Wr, br):
    # r[0] = sigmoid(r_p[0])
    return sigmoid(r_p_0(Wr, br))

def h_h_p_0(W, b):
    # h_h_p[0] = np.dot(W, x[0]) + b
    return np.dot(W, x[0]) + b

def h_h_0(W, b):
    # h_h[0] = tanh(h_h_p[0])
    return tanh(h_h_p_0(W, b))

def h_0(W, Wz, b, bz):
    # h[0] = z[0] * h_h[0]
    return z_0(Wz, bz) * h_h_0(W, b)

def z_p_1(Uz, W, Wz, b, bz):
    # z_p[1] = (np.dot(Wz, x[1]) + np.dot(Uz, h[0]) + bz)
    return (np.dot(Wz, x[1]) + np.dot(Uz, h_0(W, Wz, b, bz)) + bz)

def z_1(Uz, W, Wz, b, bz):
    # z[1] = sigmoid(z_p[1])
    return sigmoid(z_p_1(Uz, W, Wz, b, bz))

def r_p_1(Ur, W, Wr, Wz, b, br, bz):
    # r_p[1] = (np.dot(Wr, x[1]) + np.dot(Ur, h[0]) + br)
    return (np.dot(Wr, x[1]) + np.dot(Ur, h_0(W, Wz, b, bz)) + br)

def r_1(Ur, W, Wr, Wz, b, br, bz):
    # r[1] = sigmoid(r_p[1])
    return sigmoid(r_p_1(Ur, W, Wr, Wz, b, br, bz))

def h_h_p_1(U, Ur, W, Wr, Wz, b, br, bz):
    # h_h_p[1] = (np.dot(W, x[1]) + np.dot(U, r[1] * h[0]) + b)
    return (np.dot(W, x[1]) + np.dot(U, r_1(Ur, W, Wr, Wz, b, br, bz) * h_0(W, Wz, b, bz)) + b)

def h_h_1(U, Ur, W, Wr, Wz, b, br, bz):
    # h_h[1] = tanh(h_h_p[1])
    return tanh(h_h_p_1(U, Ur, W, Wr, Wz, b, br, bz))

def h_1(U, Ur, Uz, W, Wr, Wz, b, br, bz):
    # h[1] = (1 - z[1]) * h[0] + z[1] * h_h[1]
    return (1 - z_1(Uz, W, Wz, b, bz)) * h_0(W, Wz, b, bz) + z_1(Uz, W, Wz, b, bz) * h_h_1(U, Ur, W, Wr, Wz, b, br, bz)

# Dense
# y[0]
def y_h_0(V, c, W, Wz, b, bz):
    # y_h = np.dot(V, h[0]) + c
    return np.dot(V, h_0(W, Wz, b, bz)) + c

# y[1]
def y_h_1(V, c, U, Ur, Uz, W, Wr, Wz, b, br, bz):
    # y_h = np.dot(V, h[1]) + c
    return np.dot(V, h_1(U, Ur, Uz, W, Wr, Wz, b, br, bz)) + c



def gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t=None):
    if return_sequences == True and t == 0:
        return linear(y_h_0(V, c, W, Wz, b, bz))
    else: # (return_sequences == True and t == 1) or return_sequences == False
        return linear(y_h_1(V, c, U, Ur, Uz, W, Wr, Wz, b, br, bz))

"""
# GRU
z_p[0] = np.dot(Wz, x[0]) + bz
z[0] = sigmoid(np.dot(Wz, x[0]) + bz)
r_p[0] = np.dot(Wr, x[0]) + br
r[0] = sigmoid(np.dot(Wr, x[0]) + br)
h_h_p[0] = np.dot(W, x[0]) + b
h_h[0] = tanh(np.dot(W, x[0]) + b)
h[0] = sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)


z_p[1] = (np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz)
z[1] = sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz))
r_p[1] = (np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)
r[1] = sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br))
h_h_p[1] = (np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b)
h_h[1] = tanh((np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b))
h[1] = (1 - sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz))) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b) + sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz)) * tanh((np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b))

# Dense
# y[0]
y_h = np.dot(V, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + c
y[0] = linear(np.dot(V, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + c)

# y[1]
y_h = np.dot(V, (1 - sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz))) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b) + sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz)) * tanh((np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b))) + c
y[1] = linear(np.dot(V, (1 - sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz))) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b) + sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz)) * tanh((np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b))) + c)


def gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t=None):
    if return_sequences == True and t == 0:
        return linear(np.dot(V, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + c)
    else: # (return_sequences == True and t == 1) or return_sequences == False
        return linear(np.dot(V, (1 - sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz))) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b) + sigmoid((np.dot(Wz, x[1]) + np.dot(Uz, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + bz)) * tanh((np.dot(W, x[1]) + np.dot(U, sigmoid((np.dot(Wr, x[1]) + np.dot(Ur, sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + br)) * sigmoid(np.dot(Wz, x[0]) + bz) * tanh(np.dot(W, x[0]) + b)) + b))) + c)
"""


def loss(_W, y):
    if return_sequences == True:
        return np.sum((forwardprop(_W, 0) - y[0]) ** 2 / 2) + np.sum((forwardprop(_W, 1) - y[1]) ** 2 / 2)
    else:
        return np.sum((forwardprop(_W) - y) ** 2 / 2)


gradient_fun = grad(loss)

def eval_grad(_W, _dW, param):
    weights = gradient_fun(_W, y)
    print("{} = {}".format(str(param), weights))
    print("\t=> OK") if np.allclose(weights, _dW) else print("***** ERROR!!!")


print("========== Autograd ==========")

###############
# dWz, dWr, dW
###############

#
# Wz
#
def forwardprop(Wz, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(Wz, dWz, "dWz")

#
# Wr
#
def forwardprop(Wr, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(Wr, dWr, "dWr")

#
# W
#
def forwardprop(W, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(W, dW, "dW")

###############
# dUz, dUr, dU
###############

#
# Uz
#
def forwardprop(Uz, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(Uz, dUz, "dUz")

#
# Ur
#
def forwardprop(Ur, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(Ur, dUr, "dUr")

#
# U
#
def forwardprop(U, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(U, dU, "dU")

###############
# dbz, dbr, db
###############

#
# bz
#
def forwardprop(bz, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(bz, dbz, "dbz")

#
# br
#
def forwardprop(br, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(br, dbr, "dbr")

#
# b
#
def forwardprop(b, t=None):
    return gru(Wz, Wr, W, Uz, Ur, U, bz, br, b, t)

eval_grad(b, db, "db")
