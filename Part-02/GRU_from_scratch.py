#
# Sine wave prediction using GRU made from scratch.
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import numpy as np
import matplotlib.pyplot as plt
import DataSet as ds
import sys

sys.path.append("..")
from Common import Optimizer, Layers
from Common.ActivationFunctions import sigmoid, deriv_sigmoid, tanh, deriv_tanh, linear, deriv_linear
from Common.Optimizer import update_weights

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""


class GRU:
    def __init__(self, input_units, hidden_units, return_sequences=False):
        self.return_sequences = return_sequences

        self.input_units = input_units
        self.hidden_units = hidden_units

        """
        Initialize random weights and bias using Glorot
        and Orthogonal Weight Initializations.

        Glorat Weight Initialization: Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

        Orthogonal Weight Initialization: Saxe et al.,
        https://arxiv.org/pdf/1312.6120.pdf
        """
        self.Wz = np.random.randn(hidden_units, input_units) * np.sqrt(2.0 / (hidden_units + input_units))
        self.Wr = np.random.randn(hidden_units, input_units) * np.sqrt(2.0 / (hidden_units + input_units))
        self.W = np.random.randn(hidden_units, input_units) * np.sqrt(2.0 / (hidden_units + input_units))

        self.Uz = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (2 * hidden_units))
        self.Uz, _, _ = np.linalg.svd(self.Uz) # Orthogonal Weight Initialization
        self.Ur = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (2 * hidden_units))
        self.Ur, _, _ = np.linalg.svd(self.Ur) # Orthogonal Weight Initialization
        self.U = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (2 * hidden_units))
        self.U, _, _ = np.linalg.svd(self.U) # Orthogonal Weight Initialization

        self.bz = np.random.randn(hidden_units, 1) * np.sqrt(2.0 / (1 + hidden_units))
        self.br = np.random.randn(hidden_units, 1) * np.sqrt(2.0 / (1 + hidden_units))
        self.b = np.random.randn(hidden_units, 1) * np.sqrt(2.0 / (1 + hidden_units))


    def get_grads(self):
        return [self.dWz, self.dWr, self.dW,
                self.dUz, self.dUr, self.dU,
                self.dbz, self.dbr, self.db]

    def get_params(self):
        return [self.Wz, self.Wr, self.W,
                self.Uz, self.Ur, self.U,
                self.bz, self.br, self.b]

    def num_params(self):
        params = self.Wz.size + self.Wr.size + self.W.size
        params += self.Uz.size + self.Ur.size + self.U.size
        params += self.bz.size + self.br.size + self.b.size
        return params

    def forward_prop(self, x, n_sequence):
        self.x = x
        self.n_sequence = n_sequence

        self.z = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.z_p = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.r = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.r_p = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.h_h = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.h_h_p = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.h = np.zeros([self.n_sequence, self.hidden_units, 1])

        for t in range(self.n_sequence):
            if t == 0:
                self.z_p[t] = np.dot(self.Wz, self.x[t]) + self.bz
                self.z[t] = sigmoid(self.z_p[t])
                self.r_p[t] = np.dot(self.Wr, self.x[t]) + self.br
                self.r[t] = sigmoid(self.r_p[t])
                self.h_h_p[t] = np.dot(self.W, self.x[t]) + self.b
                self.h_h[t] = tanh(self.h_h_p[t])
                self.h[t] = self.z[t] * self.h_h[t]
            else:
                self.z_p[t] = (np.dot(self.Wz, self.x[t]) + np.dot(self.Uz, self.h[t - 1]) + self.bz)
                self.z[t] = sigmoid(self.z_p[t])
                self.r_p[t] = (np.dot(self.Wr, self.x[t]) + np.dot(self.Ur, self.h[t - 1]) + self.br)
                self.r[t] = sigmoid(self.r_p[t])
                self.h_h_p[t] = (np.dot(self.W, self.x[t]) + np.dot(self.U, self.r[t] * self.h[t - 1]) + self.b)
                self.h_h[t] = tanh(self.h_h_p[t])
                self.h[t] = (1 - self.z[t]) * self.h[t - 1] + self.z[t] * self.h_h[t]

        if self.return_sequences == False:
            return self.h[-1]
        else:
            return self.h


    def back_prop(self, grads):
        self.dWz = np.zeros_like(self.Wz)
        self.dWr = np.zeros_like(self.Wr)
        self.dW = np.zeros_like(self.W)

        self.dUz = np.zeros_like(self.Uz)
        self.dUr = np.zeros_like(self.Ur)
        self.dU = np.zeros_like(self.U)

        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.db = np.zeros_like(self.b)

        dh = np.zeros([self.n_sequence, self.hidden_units, 1])

        dx = np.zeros([self.n_sequence, self.input_units, 1])

        for t in reversed(range(self.n_sequence)):

            if t == self.n_sequence - 1:
                if self.return_sequences == True:
                    dh[t] = grads[t]
                else:
                    dh[t] = grads
            else:
                d1 = np.dot(self.Uz.T, dh[t + 1] * (self.h_h[t + 1] - self.h[t]) * deriv_sigmoid(self.z_p[t + 1]))
                d2 = np.dot(self.r[t + 1] * self.U.T, dh[t + 1] * self.z[t + 1] * deriv_tanh(self.h_h_p[t + 1]))
                d3 = np.dot(self.Ur.T, np.dot(self.h[t] * self.U.T, dh[t + 1] * self.z[t + 1] * deriv_tanh(self.h_h_p[t + 1])) * deriv_sigmoid(self.r_p[t + 1]))
                d4 = dh[t + 1] * (1 - self.z[t + 1])

                dh[t] = d1 + d2 + d3 + d4
                if self.return_sequences == True:
                    dh[t] += grads[t]

        for t in range(self.n_sequence):

            if t > 0:
                _dbz = (dh[t] * deriv_sigmoid(self.z_p[t]) * (self.h_h[t] - self.h[t - 1]))
            else:
                _dbz = dh[t] * deriv_sigmoid(self.z_p[t]) * self.h_h[t]

            self.dbz += _dbz
            self.dWz += np.dot(_dbz, self.x[t].T)
            if t > 0:
                self.dUz += np.dot(_dbz, self.h[t - 1].T)

            if t > 0:
                _dbr = np.dot(self.h[t - 1] * self.U.T, dh[t] * self.z[t] * deriv_tanh(self.h_h_p[t])) * deriv_sigmoid(self.r_p[t])
                self.dbr += _dbr
                self.dWr += np.dot(_dbr, self.x[t].T)
                self.dUr += np.dot(_dbr, self.h[t - 1].T)

            _db = dh[t] * self.z[t] * deriv_tanh(self.h_h_p[t])
            self.db += _db
            self.dW += np.dot(_db, self.x[t].T)
            if t > 0:
                self.dU += np.dot(_db, (self.r[t] * self.h[t - 1]).T)

            dx[t] = np.dot(self.Wz.T, _dbz) + np.dot(self.W.T, _db)
            if t > 0:
                dx[t] += np.dot(self.Wr.T, _dbr)


        return dx



def summary(gru, dense, n_sequence, hidden_units):
    params = 0
    print("_________________________________________________________________")
    print(" Layer (type)                Output Shape              Param #")
    print("=================================================================")
    params = gru.num_params()
    print(
        " gru (GRU)                   (None, {:>3},{:>3})          {:>8}".format(
            n_sequence, hidden_units, params
        )
    )
    print("")
    print(
        " dense (Dense)               (None, {:>3},{:>3})          {:>8}".format(
            n_sequence, 1, dense.num_params()
        )
    )
    print("=================================================================")
    params += dense.num_params()
    print("Total params: {}".format(params))


def plot_fig(simple_rnn, dense, wave_data, n_sample, n_sequence):

    wave = wave_data
    z = wave[0:n_sequence]
    input = wave[0:n_sequence+1]
    sin = [None for i in range(n_sequence)]
    gen = [None for i in range(n_sequence)]

    for j in range(n_sample):
        h = simple_rnn.forward_prop(z, n_sequence)
        y = dense.forward_prop(h)
        z = np.append(z, y)[1:]
        gen.append(y)
        sin.append(wave[j+n_sequence])

    plt.plot(input, color="b", label="input")
    plt.plot(sin, "--", color="#888888", label="sine wave")
    plt.plot(gen, color="r", label="predict")
    plt.title("Prediction")
    plt.legend()
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.show()


# ============================
# Create Dataset
# ============================
n_sequence = 25
n_data = 100

n_sample = n_data - n_sequence  # number of samples

sin_data = ds.create_wave(n_data, 0.05)
X, Y = ds.dataset(sin_data, n_sequence)

# ============================
# Create Model
# ============================

input_units = 1
hidden_units = 32
output_units = 1

gru = GRU(input_units, hidden_units)
dense = Layers.Dense(hidden_units, output_units, linear, deriv_linear)

summary(gru, dense, n_sequence, hidden_units)

# ============================
# Training
# ============================

def train(gru, dense, X, Y, optimizer):
    # Forward Propagation
    last_h = gru.forward_prop(X, n_sequence)
    y = dense.forward_prop(last_h)

    # Back Propagation Through Time
    loss = np.sum((y - Y) ** 2 / 2)
    dL = y - Y

    grads = dense.back_prop(dL)
    _ = gru.back_prop(grads)

    # Weights and Bias Update
    update_weights([dense, gru], optimizer=optimizer)

    return loss


history_loss = []

n_epochs = 200
lr = 0.0001

beta1 = 0.99
beta2 = 0.9999
optimizer = Optimizer.Adam(lr=lr, beta1=beta1, beta2=beta2)


for epoch in range(1, n_epochs + 1):

    loss = 0.0
    for j in range(n_sample):
        loss += train(gru, dense, X[j], Y[j], optimizer)

    history_loss.append(loss)
    if epoch % 10 == 0:
        print("epoch: {}/{}\t Loss = {:.6f}".format(epoch, n_epochs, loss))

#
#
#
plt.plot(history_loss, color="b", label="loss")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ============================
# Prediction
# ============================

sin_data = ds.create_wave(n_data, 0.0)
plot_fig(gru, dense, sin_data, n_sample, n_sequence)
