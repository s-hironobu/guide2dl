#
# LSTM Model Validation with Autograd
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
Wi = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
Wf = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
Wc = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10
Wo = np.arange(1, input_units * hidden_units + 1).reshape((hidden_units, input_units)) / 10

Ui = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10
Uf = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10
Uc = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10
Uo = np.arange(1, hidden_units * hidden_units + 1).reshape((hidden_units, hidden_units)) / 10


bi = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
bf = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
bc = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10
bo = np.arange(1, 1 * hidden_units + 1).reshape((hidden_units, 1)) / 10

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
h = np.zeros([n_sequence, hidden_units, 1])
C = np.zeros([n_sequence, hidden_units, 1])
i = np.zeros([n_sequence, hidden_units, 1])
i_p = np.zeros([n_sequence, hidden_units, 1])
c_h = np.zeros([n_sequence, hidden_units, 1])
c_h_p = np.zeros([n_sequence, hidden_units, 1])
f = np.zeros([n_sequence, hidden_units, 1])
f_p = np.zeros([n_sequence, hidden_units, 1])
o = np.zeros([n_sequence, hidden_units, 1])
o_p = np.zeros([n_sequence, hidden_units, 1])

# LSTM

for t in range(n_sequence):
    if t == 0:
        f_p[t] = np.dot(Wf, x[t]) + bf
        i_p[t] = np.dot(Wi, x[t]) + bi
        c_h_p[t] = np.dot(Wc, x[t]) + bc
        o_p[t] = np.dot(Wo, x[t]) + bo
    else:
        f_p[t] = np.dot(Wf, x[t]) + np.dot(Uf, h[t - 1]) + bf
        i_p[t] = np.dot(Wi, x[t]) + np.dot(Ui, h[t - 1]) + bi
        c_h_p[t] = np.dot(Wc, x[t]) + np.dot(Uc, h[t - 1]) + bc
        o_p[t] = np.dot(Wo, x[t]) + np.dot(Uo, h[t - 1]) + bo

    f[t] = sigmoid(f_p[t])
    i[t] = sigmoid(i_p[t])
    c_h[t] = tanh(c_h_p[t])
    o[t] = sigmoid(o_p[t])
    C[t] = i[t] * c_h[t] if t == 0 else i[t] * c_h[t] + f[t] * C[t - 1]
    h[t] = o[t] * tanh(C[t])


# Dense
if return_sequences == True:
    y_h = np.zeros([n_sequence, output_units, 1])
    for s in range(n_sequence):
        y_h[s] = np.dot(V, h[s]) + c
    _y = linear(y_h)
else:
    y_h = np.dot(V, h[-1]) + c
    _y = linear(y_h)


################################################
# BPTT
################################################

dWi = np.zeros_like(Wi)
dWf = np.zeros_like(Wf)
dWc = np.zeros_like(Wc)
dWo = np.zeros_like(Wo)

dUi = np.zeros_like(Ui)
dUf = np.zeros_like(Uf)
dUc = np.zeros_like(Uc)
dUo = np.zeros_like(Uo)

dbi = np.zeros_like(bi)
dbf = np.zeros_like(bf)
dbc = np.zeros_like(bc)
dbo = np.zeros_like(bo)

dV = np.zeros_like(V)
dc = np.zeros_like(c)

dh = np.zeros([n_sequence, hidden_units, 1])
dC = np.zeros([n_sequence, hidden_units, 1])

dL = _y - y

# Dense
if return_sequences == True:
    dc = deriv_linear(y_h) * dL
    grads = np.zeros([n_sequence, hidden_units, 1])
    for s in range(n_sequence):
        grads[s] = np.dot(V.T, dc[s])
else:
    dc = deriv_linear(y_h[-1]) * dL
    grads = np.dot(V.T, dc)

# LSTM

for t in reversed(range(n_sequence)):

    if t == n_sequence - 1:
        if return_sequences == True:
            dh[t] = grads[t]
        else:
            dh[t] = grads

        dC[t] = dh[t] * o[t] * deriv_tanh(C[t])
    else:
        d1 = np.dot(Uo.T, dh[t + 1] * tanh(C[t + 1]) * deriv_sigmoid(o_p[t + 1]))
        d2 = np.dot(Ui.T, dh[t + 1] * o[t + 1] * deriv_tanh(C[t + 1]) * c_h[t + 1] * deriv_sigmoid(i_p[t + 1]))
        d3 = np.dot(Uc.T, dh[t + 1] * o[t + 1] * deriv_tanh(C[t + 1]) * i[t + 1] * deriv_tanh(c_h_p[t + 1]))
        d4 = np.dot(Uf.T, dh[t + 1] * o[t + 1] * deriv_tanh(C[t + 1]) * C[t] * deriv_sigmoid(f_p[t + 1]))

        dh[t] = d1 + d2 + d3 + d4
        if return_sequences == True:
            dh[t] += grads[t]

        dC[t] = dh[t] * o[t] * deriv_tanh(C[t]) + dC[t + 1] * f[t + 1]

for t in range(n_sequence):

    _do = dh[t] * tanh(C[t]) * deriv_sigmoid(o_p[t])
    dbo += _do
    dWo += np.dot(_do, x[t].T)
    if t > 0:
        dUo += np.dot(_do, h[t - 1].T)

    _di = dC[t] * c_h[t] * deriv_sigmoid(i_p[t])
    dbi += _di
    dWi += np.dot(_di, x[t].T)
    if t > 0:
        dUi += np.dot(_di, h[t - 1].T)

    _dc = dC[t] * i[t] * deriv_tanh(c_h_p[t])
    dbc += _dc
    dWc += np.dot(_dc, x[t].T)
    if t > 0:
        dUc += np.dot(_dc, h[t - 1].T)

    if t > 0:
        _df = dC[t] * C[t - 1] * deriv_sigmoid(f_p[t])
        dbf += _df
        dWf += np.dot(_df, x[t].T)
        dUf += np.dot(_df, h[t - 1].T)


print("========== BPTT ==========")

print("dWo =", dWo)
print("dWi =", dWi)
print("dWc =", dWc)
print("dWf =", dWf)
print("dUo =", dUo)
print("dUi =", dUi)
print("dUc =", dUc)
print("dUf =", dUf)
print("dbo =", dbo)
print("dbi =", dbi)
print("dbc =", dbc)
print("dbf =", dbf)

################################################
# Validate
################################################

"""
# LSTM
f_p[0] = (np.dot(Wf, x[0]) + bf)
i_p[0] = (np.dot(Wi, x[0]) + bi)
c_h_p[0] = (np.dot(Wc, x[0]) + bc)
o_p[0] = (np.dot(Wo, x[0]) + bo)
f[0] = sigmoid(f_p[0])
i[0] = sigmoid(i_p[0])
c_h[0] = tanh(c_h_p[0])
o[0] = sigmoid(o_p[0])
C[0] = (i[0] * c_h[0])
h[0] = (o[0] * tanh(C[0]))

f_p[1] = (np.dot(Wf, x[1]) + np.dot(Uf, h[0]) + bf)
i_p[1] = (np.dot(Wi, x[1]) + np.dot(Ui, h[0]) + bi)
c_h_p[1] = (np.dot(Wc, x[1]) + np.dot(Uc, h[0]) + bc)
o_p[1] = (np.dot(Wo, x[1]) + np.dot(Uo, h[0]) + bo)

f[1] = sigmoid(f_p[1])
i[1] = sigmoid(i_p[1])
c_h[1] = tanh(c_h_p[1])
o[1] = sigmoid(o_p[1])
C[1] = (i[1] * c_h[1] + f[1] * C[0])
h[1] = (o[1] * tanh(C[1]))

# Dense
# y[0]
y_h = np.dot(V, h[0]) + c
_y = linear(y_h)

# y[1]
y_h = np.dot(V, h[1]) + c
_y = linear(y_h)
"""

# LSTM
def f_p_0(Wf, bf):
    # f_p[0] = (np.dot(Wf, x[0]) + bf)
    return np.dot(Wf, x[0]) + bf


def i_p_0(Wi, bi):
    # i_p[0] = (np.dot(Wi, x[0]) + bi)
    return np.dot(Wi, x[0]) + bi


def c_h_p_0(Wc, bc):
    # c_h_p[0] = (np.dot(Wc, x[0]) + bc)
    return np.dot(Wc, x[0]) + bc


def o_p_0(Wo, bo):
    # o_p[0] = (np.dot(Wo, x[0]) + bo)
    return np.dot(Wo, x[0]) + bo


def f_0(Wf, bf):
    # f[0] = sigmoid(f_p[0])
    return sigmoid(f_p_0(Wf, bf))


def i_0(Wi, bi):
    # i[0] = sigmoid(i_p[0])
    return sigmoid(i_p_0(Wi, bi))


def c_h_0(Wc, bc):
    # c_h[0] = tanh(c_h_p[0])
    return tanh(c_h_p_0(Wc, bc))


def o_0(Wo, bo):
    # o[0] = sigmoid(o_p[0])
    return sigmoid(o_p_0(Wo, bo))


def C_0(Wc, Wi, bc, bi):
    # C[0] = (i[0] * c_h[0])
    return i_0(Wi, bi) * c_h_0(Wc, bc)


def h_0(Wc, Wi, Wo, bc, bi, bo):
    # h[0] = (o[0] * tanh(C[0]))
    return o_0(Wo, bo) * tanh(C_0(Wc, Wi, bc, bi))


def f_p_1(Uf, Wc, Wf, Wi, Wo, bc, bf, bi, bo):
    # f_p[1] = (np.dot(Wf, x[1]) + np.dot(Uf, h[0]) + bf)
    return np.dot(Wf, x[1]) + np.dot(Uf, h_0(Wc, Wi, Wo, bc, bi, bo)) + bf


def i_p_1(Ui, Wc, Wi, Wo, bc, bi, bo):
    # i_p[1] = (np.dot(Wi, x[1]) + np.dot(Ui, h[0]) + bi)
    return np.dot(Wi, x[1]) + np.dot(Ui, h_0(Wc, Wi, Wo, bc, bi, bo)) + bi


def c_h_p_1(Uc, Wc, Wi, Wo, bc, bi, bo):
    # c_h_p[1] = (np.dot(Wc, x[1]) + np.dot(Uc, h[0]) + bc)
    return np.dot(Wc, x[1]) + np.dot(Uc, h_0(Wc, Wi, Wo, bc, bi, bo)) + bc


def o_p_1(Uo, Wc, Wi, Wo, bc, bi, bo):
    # o_p[1] = (np.dot(Wo, x[1]) + np.dot(Uo, h[0]) + bo)
    return np.dot(Wo, x[1]) + np.dot(Uo, h_0(Wc, Wi, Wo, bc, bi, bo)) + bo


def f_1(Uf, Wc, Wf, Wi, Wo, bc, bf, bi, bo):
    # f[1] = sigmoid(f_p[1])
    return sigmoid(f_p_1(Uf, Wc, Wf, Wi, Wo, bc, bf, bi, bo))


def i_1(Ui, Wc, Wi, Wo, bc, bi, bo):
    # i[1] = sigmoid(i_p[1])
    return sigmoid(i_p_1(Ui, Wc, Wi, Wo, bc, bi, bo))


def c_h_1(Uc, Wc, Wi, Wo, bc, bi, bo):
    # c_h[1] = tanh(c_h_p[1])
    return tanh(c_h_p_1(Uc, Wc, Wi, Wo, bc, bi, bo))


def o_1(Uo, Wc, Wi, Wo, bc, bi, bo):
    # o[1] = sigmoid(o_p[1])
    return sigmoid(o_p_1(Uo, Wc, Wi, Wo, bc, bi, bo))


def C_1(Uc, Uf, Ui, Wc, Wf, Wi, Wo, bc, bf, bi, bo):
    # C[1] = (i[1] * c_h[1] + f[1] * C[0])
    return i_1(Ui, Wc, Wi, Wo, bc, bi, bo) * c_h_1(Uc, Wc, Wi, Wo, bc, bi, bo) + f_1(Uf, Wc, Wf, Wi, Wo, bc, bf, bi, bo) * C_0(Wc, Wi, bc, bi)


def h_1(Uc, Uf, Ui, Uo, Wc, Wf, Wi, Wo, bc, bf, bi, bo):
    # h[1] = (o[1] * tanh(C[1]))
    return o_1(Uo, Wc, Wi, Wo, bc, bi, bo) * tanh(C_1(Uc, Uf, Ui, Wc, Wf, Wi, Wo, bc, bf, bi, bo))


# Dense
# y[0]
def y_h_0(V, c, Wc, Wi, Wo, bc, bi, bo):
    # y_h = np.dot(V, h[0]) + c
    return np.dot(V, h_0(Wc, Wi, Wo, bc, bi, bo)) + c


# y[1]
def y_h_1(V, c, Uc, Uf, Ui, Uo, Wc, Wf, Wi, Wo, bc, bf, bi, bo):
    # y_h = np.dot(V, h[1]) + c
    return np.dot(V, h_1(Uc, Uf, Ui, Uo, Wc, Wf, Wi, Wo, bc, bf, bi, bo)) + c


def lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t=None):
    if return_sequences == True and t == 0:
        return linear(y_h_0(V, c, Wc, Wi, Wo, bc, bi, bo))
    else:  # (return_sequences == True and t == 1) or return_sequences == False
        return linear(y_h_1(V, c, Uc, Uf, Ui, Uo, Wc, Wf, Wi, Wo, bc, bf, bi, bo))


"""
# LSTM
f_p[0] = (np.dot(Wf, x[0]) + bf)
i_p[0] = (np.dot(Wi, x[0]) + bi)
c_h_p[0] = (np.dot(Wc, x[0]) + bc)
o_p[0] = (np.dot(Wo, x[0]) + bo)
f[0] = sigmoid(np.dot(Wf, x[0]) + bf)
i[0] = sigmoid(np.dot(Wi, x[0]) + bi)
c_h[0] = tanh(np.dot(Wc, x[0]) + bc)
o[0] = sigmoid(np.dot(Wo, x[0]) + bo)
C[0] = (sigmoid(np.dot(Wi, x[0]) + bi) * tanh(np.dot(Wc, x[0]) + bc))
h[0] = sigmoid(np.dot(Wo, x[0]) + bo) * tanh(sigmoid((np.dot(Wi, x[0]) + bi) * tanh((np.dot(Wc, x[0]) + bc))))

f_p[1] = (np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)
i_p[1] = (np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)
c_h_p[1] = (np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)
o_p[1] = (np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo)

f[1] = sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf))
i[1] = sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi))
c_h[1] = tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc))
o[1] = sigmoid((np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo))
C[1] = (sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)) * tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)) + sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)) * (sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc))))
h[1] = (sigmoid((np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo)) * tanh((sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)) * tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)) + sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)) * (sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc))))))

# Dense
# y[0]
y_h = np.dot(V, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + c
y[0] = linear(np.dot(V, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + c)

# y[1]
y_h = np.dot(V, (sigmoid((np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo)) * tanh((sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)) * tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)) + sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)) * (sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc))))))) + c
y[1] = linear(np.dot(V, (sigmoid((np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo)) * tanh((sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)) * tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)) + sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)) * (sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc))))))) + c)

def lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t=None):
    if return_sequences == True and t == 0:
        return linear(np.dot(V, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + c)
    else: # (return_sequences == True and t == 1) or return_sequences == False
        return linear(np.dot(V, (sigmoid((np.dot(Wo, x[1]) + np.dot(Uo, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bo)) * tanh((sigmoid((np.dot(Wi, x[1]) + np.dot(Ui, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bi)) * tanh((np.dot(Wc, x[1]) + np.dot(Uc, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bc)) + sigmoid((np.dot(Wf, x[1]) + np.dot(Uf, (sigmoid((np.dot(Wo, x[0]) + bo)) * tanh((sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc)))))) + bf)) * (sigmoid((np.dot(Wi, x[0]) + bi)) * tanh((np.dot(Wc, x[0]) + bc))))))) + c)
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
# dWo, dWi, dWc, dWf
###############

#
# Wo
#
def forwardprop(Wo, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Wo, dWo, "dWo")

#
# Wi
#
def forwardprop(Wi, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Wi, dWi, "dWi")

#
# Wc
#
def forwardprop(Wc, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Wc, dWc, "dWc")

#
# Wf
#
def forwardprop(Wf, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Wf, dWf, "dWf")

###############
# dUo, dUi, dUc, dUf
###############

#
# Uo
#
def forwardprop(Uo, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Uo, dUo, "dUo")

#
# Ui
#
def forwardprop(Ui, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Ui, dUi, "dUi")

#
# Uc
#
def forwardprop(Uc, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Uc, dUc, "dUc")

#
# Uf
#
def forwardprop(Uf, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(Uf, dUf, "dUf")

###############
# dbo, dbi, dbc, dbf
###############

#
# bo
#
def forwardprop(bo, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(bo, dbo, "dbo")

#
# bi
#
def forwardprop(bi, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(bi, dbi, "dbi")

#
# bc
#
def forwardprop(bc, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(bc, dbc, "dbc")

#
# bf
#
def forwardprop(bf, t=None):
    return lstm(Wo, Wi, Wc, Wf, Uo, Ui, Uc, Uf, bo, bi, bc, bf, t)


eval_grad(bf, dbf, "dbf")
