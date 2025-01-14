#
# Compare optimizers GD and ADAM.
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
from Common import Optimizer, Layers
from Common.Optimizer import update_weights
from Common.ActivationFunctions import sigmoid, deriv_sigmoid


# np.random.seed(0)

# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([[0], [1], [1], [0]])

# Convert row vector into column vector.
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 1, 1)

# ========================================
# Common parameters
# ========================================

input_nodes = 2
hidden_nodes = 3
output_nodes = 1

n_epochs = 10000  # Epochs
lr = 0.1  # Learning rate


# --------------------------------------------------
# Gradient descent
# --------------------------------------------------

# ========================================
# Create Model
# ========================================

dense = Layers.Dense(input_nodes, hidden_nodes, sigmoid, deriv_sigmoid)
dense_1 = Layers.Dense(hidden_nodes, output_nodes, sigmoid, deriv_sigmoid)

#
# Show parameters
#
params = 0
print("_________________________________________________________________")
print(" Layer (type)                Output Shape              Param #")
print("=================================================================")
param = dense.num_params()
params += param
print(
    " dense (Dense)               (None, {:>2})             {:>8}".format(
        hidden_nodes, param
    )
)
param = dense_1.num_params()
params += param
print(
    " dense_1 (Dense)             (None, {:>2})             {:>8}".format(
        output_nodes, param
    )
)
print("=================================================================")
print("Total params: {}\n".format(params))


# ========================================
# Training
# ========================================
print("========= Gradient Descent =========")


def train(x, Y, lr=0.001):

    # Forward Propagation
    y = dense.forward_prop(x)
    y = dense_1.forward_prop(y)

    # Back Propagation
    loss = np.sum((y - Y) ** 2 / 2)
    dL = y - Y

    dx = dense_1.back_prop(dL)
    _ = dense.back_prop(dx)

    # Weights and Bias Update
    update_weights([dense, dense_1], lr=lr)

    return loss


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


# ========================================
# Test
# ========================================

print("------------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(0, len(Y)):
    x = X[i]
    y = dense.forward_prop(x)
    y = dense_1.forward_prop(y)
    print(" {} XOR  {} => {:.4f}".format(X[i][0][0], X[i][1][0], y[0][0]))

print("========================")


# --------------------------------------------------
# ADAM
# --------------------------------------------------

print("=========       ADAM       =========")

# ========================================
# Recreate Model
# ========================================

del dense
del dense_1

dense = Layers.Dense(input_nodes, hidden_nodes, sigmoid, deriv_sigmoid)
dense_1 = Layers.Dense(hidden_nodes, output_nodes, sigmoid, deriv_sigmoid)

# ========================================
# Training
# ========================================


def train_adam(x, Y, optimizer):

    # Forward Propagation
    y = dense.forward_prop(x)
    y = dense_1.forward_prop(y)

    # Back Propagation
    loss = np.sum((y - Y) ** 2 / 2)
    dL = y - Y

    dx = dense_1.back_prop(dL)
    _ = dense.back_prop(dx)

    # Update weights
    update_weights([dense, dense_1], optimizer=optimizer)

    return loss


lr = 0.1
beta1 = 0.9
beta2 = 0.999
optimizer = Optimizer.Adam(lr=lr, beta1=beta1, beta2=beta2)

history_loss_adam = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):

    loss = 0.0

    for i in range(0, len(Y)):
        loss += train_adam(X[i], Y[i], optimizer)

    history_loss_adam.append(loss / len(Y))
    if epoch % 1000 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, loss))


# ========================================
# Test
# ========================================

print("------------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(0, len(Y)):
    x = X[i]
    y = dense.forward_prop(x)
    y = dense_1.forward_prop(y)
    print(" {} XOR  {} => {:.4f}".format(X[i][0][0], X[i][1][0], y[0][0]))

print("========================")


# --------------------------------------------------
# Show loss history
# --------------------------------------------------
plt.plot(history_loss_gd, color="b", label="Gradient descent")
plt.plot(history_loss_adam, color="r", label="adam")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
