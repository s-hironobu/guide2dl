#
# Multi-Hidden Residual-Connection Layer Neural Network for learning XOR gate
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
import sys

sys.path.append("..")
from Common import Layers, Optimizer
from Common.ActivationFunctions import sigmoid, deriv_sigmoid
from Common.Optimizer import update_weights

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""

# ========================================
# Input number of residual connected hidden layers
# ========================================

print("Input the number of residual connected hidden layers (1 to 10): ", end="")
val = input()

val = int(val)
if val < 1 or 10 < val:
    print("Error: {} is out of range.".format(val))
    sys.exit()

NumLayers = val

# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([0, 1, 1, 0])

# Convert row vectors into column vectors.
X = X.reshape(4, 2, 1)

# ========================================
# Create Model
# ========================================


class ResidualConnection:
    def __init__(self, node_size, activate_func, deriv_activate_func):
        self.dense = Layers.Dense(
            node_size, node_size, activate_func, deriv_activate_func
        )

    def get_grads(self):
        return self.dense.get_grads()

    def get_params(self):
        return self.dense.get_params()

    def num_params(self):
        return self.dense.num_params()

    def forward_prop(self, x):
        return x + self.dense.forward_prop(x)

    def back_prop(self, grads):
        return grads + self.dense.back_prop(grads)


input_nodes = 2
hidden_nodes = 4
output_nodes = 1

dense_in = Layers.Dense(input_nodes, hidden_nodes, sigmoid, deriv_sigmoid)

hidden_layers = []
for i in range(NumLayers):
    hidden_layers.append(ResidualConnection(hidden_nodes, sigmoid, deriv_sigmoid))

dense_out = Layers.Dense(hidden_nodes, output_nodes, sigmoid, deriv_sigmoid)

lr = 0.001  # Learning rate
beta1 = 0.99
beta2 = 0.9999
optimizer = Optimizer.Adam(lr=lr, beta1=beta1, beta2=beta2)


#
# Show parameters
#
params = 0
print("_________________________________________________________________")
print(" Layer (type)                Output Shape              Param #")
print("=================================================================")
param = dense_in.num_params()
params += param
print(
    " dense_in (Dense)            (None, {:>2})             {:>8}".format(
        hidden_nodes, param
    )
)

for i in range(NumLayers):
    param = hidden_layers[i].num_params()
    params += param
    print(
        " residual_{} (Dense)          (None, {:>2})             {:>8}".format(
            i, hidden_nodes, param
        )
    )

param = dense_out.num_params()
params += param
print(
    " dense_out (Dense)           (None, {:>2})             {:>8}".format(
        output_nodes, param
    )
)
print("=================================================================")
print("Total params: {}\n".format(params))


# ========================================
# Training
# ========================================

layers = []
layers.append(dense_in)
for i in range(NumLayers):
    layers.append(hidden_layers[i])
layers.append(dense_out)


def train(x, Y, lr=0.001, optimizer=None, max_norm=None):

    # Forward Propagation
    y = dense_in.forward_prop(x)

    for i in range(NumLayers):
        y = hidden_layers[i].forward_prop(y)

    y = dense_out.forward_prop(y)

    # Back Propagation
    loss = (y[0][0] - Y) ** 2 / 2
    dL = y[0][0] - Y

    dx = dense_out.back_prop(dL)
    for i in reversed(range(NumLayers)):
        dx = hidden_layers[i].back_prop(dx)
    _ = dense_in.back_prop(dx)

    # Weights and Bias Update
    update_weights(layers, optimizer=optimizer, max_norm=max_norm)

    return loss


n_epochs = 100000  # Epochs

history_loss_gd = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):

    loss = 0.0

    for i in range(0, len(Y)):
        loss += train(X[i], Y[i], lr, optimizer)

    history_loss_gd.append(loss / len(Y))
    if epoch % 1000 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, loss))

    if loss <= 1e-7:
        break

#
# Show loss history
#
plt.plot(history_loss_gd, color="b", label="Gradient descent")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ========================================
# Test
# ========================================

print("------------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(0, len(Y)):
    x = X[i]
    y = dense_in.forward_prop(x)

    for j in range(NumLayers):
        y = hidden_layers[j].forward_prop(y)

    y = dense_out.forward_prop(y)
    print(" {} XOR  {} => {:.4f}".format(X[i][0][0], X[i][1][0], y[0][0]))

print("========================")
