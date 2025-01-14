#
# Single Hidden Layer Neural Network for learning XOR gate from scratch
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
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from Common import Layers
from Common.Optimizer import update_weights
from Common.ActivationFunctions import sigmoid, deriv_sigmoid

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""

# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([0, 1, 1, 0])

# Convert row vectors into column vectors.
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 1, 1)

# ========================================
# Create Model
# ========================================

input_nodes = 2
hidden_nodes = 3
output_nodes = 1

dense_1 = Layers.Dense(input_nodes, hidden_nodes, sigmoid, deriv_sigmoid)
dense_2 = Layers.Dense(hidden_nodes, output_nodes, sigmoid, deriv_sigmoid)

#
# Show parameters
#
params = 0
print("_________________________________________________________________")
print(" Layer (type)                Output Shape              Param #")
print("=================================================================")
param = dense_1.num_params()
params += param
print(
    " dense_1 (Dense)             (None, {:>2})             {:>8}".format(
        hidden_nodes, param
    )
)
param = dense_2.num_params()
params += param
print(
    " dense_2 (Dense)             (None, {:>2})             {:>8}".format(
        output_nodes, param
    )
)
print("=================================================================")
print("Total params: {}\n".format(params))


# ========================================
# Training
# ========================================


def train(x, Y, lr=0.001):

    # Forward Propagation
    y = dense_1.forward_prop(x)
    y = dense_2.forward_prop(y)

    # Back Propagation
    loss = np.sum((y - Y) ** 2 / 2)
    dL = y - Y

    dx = dense_2.back_prop(dL)
    _ = dense_1.back_prop(dx)

    # Weights and Bias Update
    update_weights([dense_1, dense_2], lr=lr)

    return loss


n_epochs = 15000  # Epochs
lr = 0.1  # Learning rate

history_loss_gd = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):

    loss = 0.0

    for i in range(0, len(Y)):
        loss += train(X[i], Y[i], lr)

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
    y = dense_1.forward_prop(x)
    y = dense_2.forward_prop(y)
    print(" {} XOR  {} => {:.4f}".format(x[0][0], x[1][0], y[0][0]))

print("========================")
