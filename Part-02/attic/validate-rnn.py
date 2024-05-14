#
# Simple RNN Model Validation with Autograd
#
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import autograd.numpy as np
from autograd import grad

return_sequences = False

input_units = 1
hidden_units = 3
output_units = 1

#
# Define functions
#
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
W = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
b = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
U = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10

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

# RNN
h = np.zeros([n_sequence, hidden_units, 1])
h_h = np.zeros([n_sequence, hidden_units, 1])

for t in range(n_sequence):
    h_h[t] = np.dot(W, x[t]) + b
    if t > 0:
        h_h[t] += np.dot(U, tanh(h_h[t - 1]))
    h[t] = tanh(h_h[t])

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

dV = np.zeros_like(V)
dc = np.zeros_like(c)
dW = np.zeros_like(W)
db = np.zeros_like(b)
dU = np.zeros_like(U)
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


# RNN

for t in reversed(range(n_sequence)):

    if t == n_sequence - 1:
        if return_sequences == True:
            dh[t] = grads[t]
        else:
            dh[t] = grads
    else:
        dh[t] = np.dot(U.T, dh[t + 1] * deriv_tanh(h_h[t + 1]))
        if return_sequences == True:
            dh[t] += grads[t]

for t in range(n_sequence):

    _db = dh[t] * deriv_tanh(h_h[t])
    db += _db
    dW += np.dot(_db, x[t].T)
    if t > 0:
        dU += np.dot(_db, h[t - 1].T)

print("========== BPTT ==========")

print("dW =", dW)
print("dU =", dU)
print("db =", db)

################################################
# Validate
################################################



"""
# RNN
h_h[0] = np.dot(W, x[0]) + b
h[0] = tanh(h_h[0])

h_h[1] = np.dot(W, x[1]) + b + np.dot(U, tanh(h_h[0]))
h[1] = tanh(h_h[1])

# Dense
# y[0]
y_h = np.dot(V, h[0]) + c
_y = linear(y_h)

# y[1]
y_h = np.dot(V, h[1]) + c
_y = linear(y_h)
"""

def h_h_0(W, b):
    # h_h[0] = np.dot(W, x[0]) + b
    return np.dot(W, x[0]) + b

def h_0(W, b):
    # h[0] = tanh(h_h[0])
    return tanh(h_h_0(W, b))

def h_h_1(U, W, b):
    # h_h[1] = np.dot(W, x[1]) + b + np.dot(U, tanh(h_h[0]))
    return np.dot(W, x[1]) + b + np.dot(U, tanh(h_h_0(W, b)))

def h_1(U, W, b):
    # h[1] = tanh(h_h[1])
    return tanh(h_h_1(U, W, b))

# Dense
def y_h_0(V, c, W, b):
    # y_h = np.dot(V, h[0]) + c
    return np.dot(V, h_0(W, b)) + c

def y_h_1(V, c, U, W, b):
    # y_h = np.dot(V, h[1]) + c
    return np.dot(V, h_1(U, W, b)) + c


def rnn(W, U, b, t=None):
    if return_sequences == True and t == 0:
        return linear(y_h_0(V, c, W, b))
    else: # (return_sequences == True and t == 1) or return_sequences == False
        return linear(y_h_1(V, c, U, W, b))


"""
# RNN
h_h[0] = np.dot(W, x[0]) + b
h[0] = tanh(np.dot(W, x[0]) + b)

h_h[1] = np.dot(W, x[1]) + b + np.dot(U, tanh(np.dot(W, x[0]) + b))
h[1] = tanh(np.dot(W, x[1]) + b + np.dot(U, tanh(np.dot(W, x[0]) + b)))

# Dense
# y[0]
y_h = np.dot(V, tanh(np.dot(W, x[0]) + b)) + c
y[0] = linear(np.dot(V, tanh(np.dot(W, x[0]) + b)) + c)

# y[1]
y_h = np.dot(V, tanh(np.dot(W, x[1]) + b + np.dot(U, tanh(np.dot(W, x[0]) + b)))) + c
y[1] = linear(np.dot(V, tanh(np.dot(W, x[1]) + b + np.dot(U, tanh(np.dot(W, x[0]) + b)))) + c)

def rnn(W, U, b, t=None):
    if return_sequences == True and t == 0:
        return linear(np.dot(V, tanh(np.dot(W, x[0]) + b)) + c)
    else: # (return_sequences == True and t == 1) or return_sequences == False
        return linear(np.dot(V, tanh((np.dot(W, x[1]) + b) + np.dot(U, tanh(np.dot(W, x[0]) + b)))) + c)
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
# W, U, b
###############

#
# W
#
def forwardprop(W, t=None):
    return rnn(W, U, b, t)

eval_grad(W, dW, "dW")

#
# U
#
def forwardprop(U, t=None):
    return rnn(W, U, b, t)

eval_grad(U, dU, "dU")

#
# b
#
def forwardprop(b, t=None):
    return rnn(W, U, b, t)

eval_grad(b, db, "db")
