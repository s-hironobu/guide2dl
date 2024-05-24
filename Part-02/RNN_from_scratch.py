#
# Sine wave prediction using Simple-RNN made from scratch.
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
import matplotlib.pyplot as plt
import DataSet as ds
import sys

sys.path.append("..")
from Common import Optimizer, Layers
from Common.ActivationFunctions import tanh, deriv_tanh, linear, deriv_linear
from Common.Optimizer import update_weights

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""



class SimpleRNN:
    def __init__(self, input_units, hidden_units, activate_func, deriv_activate_func, return_sequences=False):
        self.input_units = input_units
        self.hidden_units = hidden_units

        self.activate_func = activate_func
        self.deriv_activate_func = deriv_activate_func

        self.return_sequences = return_sequences

        """
        Initialize random weights and bias using Glorot
        and Orthogonal Weight Initializations.

        Glorat Weight Initialization: Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

        Orthogonal Weight Initialization: Saxe et al.,
        https://arxiv.org/pdf/1312.6120.pdf
        """
        self.W = np.random.randn(hidden_units, input_units) * np.sqrt(2.0 / (hidden_units + input_units))
        self.b = np.random.randn(hidden_units, 1) * np.sqrt(2.0 / (hidden_units + 1))
        self.U = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (hidden_units + hidden_units))
        self.U, _, _ = np.linalg.svd(self.U) # Orthogonal Weight Initialization

    def get_grads(self):
        return [self.dW, self.dU, self.db]

    def get_params(self):
        return [self.W, self.U, self.b]

    def num_params(self):
        return self.W.size + self.b.size + self.U.size

    def forward_prop(self, x, n_sequence):
        self.x = x
        self.n_sequence = n_sequence

        self.h = np.zeros([self.n_sequence, self.hidden_units, 1])
        self.h_h = np.zeros([self.n_sequence, self.hidden_units, 1])

        for t in range(self.n_sequence):
            self.h_h[t] = np.dot(self.W, x[t]) + self.b
            if t > 0:
                self.h_h[t] += np.dot(self.U, self.activate_func(self.h_h[t - 1]))
            self.h[t] = self.activate_func(self.h_h[t])

        if self.return_sequences == False:
            return self.h[-1]
        else:
            return self.h

    def back_prop(self, grads):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dU = np.zeros_like(self.U)

        dh = np.zeros([self.n_sequence, self.hidden_units, 1])
        dx = np.zeros([self.n_sequence, self.input_units, 1])

        for t in reversed(range(self.n_sequence)):

            if t == self.n_sequence - 1:
                if self.return_sequences == True:
                    dh[t] = grads[t]
                else:
                    dh[t] = grads
            else:
                dh[t] = np.dot(self.U.T, dh[t + 1] * self.deriv_activate_func(self.h_h[t + 1]))
                if self.return_sequences == True:
                    dh[t] += grads[t]

        for t in range(self.n_sequence):

            _db = dh[t] * self.deriv_activate_func(self.h_h[t])
            self.db += _db
            self.dW += np.dot(_db, self.x[t].T)
            if t > 0:
                self.dU += np.dot(_db, self.h[t - 1].T)

            dx[t] = np.dot(self.W.T, _db)


        return dx


def summary(simple_rnn, dense, n_sequence, hidden_units):
    params = 0
    print("_________________________________________________________________")
    print(" Layer (type)                Output Shape              Param #")
    print("=================================================================")
    param = simple_rnn.num_params()
    params += param
    print(
        " simple_rnn (SimpleRNN)      (None, {:>3},{:>3})          {:>8}".format(
            n_sequence, hidden_units, param
        )
    )
    param = dense.num_params()
    params += param
    print(
        " dense (Dense)               (None, {:>3},{:>3})          {:>8}".format(
            n_sequence, output_units, param
        )
    )
    print("=================================================================")
    print("Total params: {}\n".format(params))


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
        gen.append(y[0][0])
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
# Create dataset
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

dense = Layers.Dense(hidden_units, output_units, linear, deriv_linear)
simple_rnn = SimpleRNN(input_units, hidden_units, tanh, deriv_tanh)

summary(simple_rnn, dense, n_sequence, hidden_units)

# ============================
# Training
# ============================


def train(simple_rnn, dense, X, Y, optimizer):
    #
    # Forward Propagation
    #
    last_h = simple_rnn.forward_prop(X, n_sequence)
    y = dense.forward_prop(last_h)

    #
    # Back Propagation Through Time
    #
    loss = np.sum((y - Y) ** 2 / 2)
    dL = y - Y

    grads = dense.back_prop(dL)
    _ = simple_rnn.back_prop(grads)

    # Weights and Bias Update
    update_weights([dense, simple_rnn], optimizer=optimizer)

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
        loss += train(simple_rnn, dense, X[j], Y[j], optimizer)

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
plot_fig(simple_rnn, dense, sin_data, n_sample, n_sequence)
