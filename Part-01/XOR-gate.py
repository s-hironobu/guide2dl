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
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
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
Y = np.array([[0], [1], [1], [0]])

# Convert row vectors into column vectors.
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 1, 1)

# ========================================
# Create Model
# ========================================

#
# Define numbers of input, hidden and output nodes
#
input_nodes = 2
hidden_nodes = 3
output_nodes = 1

#
# Initialize random weights and bias
#
W = np.random.uniform(size=(hidden_nodes, input_nodes))
b = np.random.uniform(size=(hidden_nodes, 1))
U = np.random.uniform(size=(output_nodes, hidden_nodes))
c = np.random.uniform(size=(output_nodes, 1))

#
# Show parameters
#
params = 0
print("_________________________________________________________________")
print(" Layer (type)                Output Shape              Param #")
print("=================================================================")
param = W.size + b.size
params += param
print(
    " dense (Dense)               (None, {:>2})             {:>8}".format(
        hidden_nodes, param
    )
)
param = U.size + c.size
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

n_epochs = 15000  # Epochs
lr = 0.1  # Learning rate

history_loss_gd = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):

    loss = 0.0

    for i in range(0, len(Y)):
        #
        # Forward Propagation
        #
        h_h = np.dot(W, X[i]) + b
        h = sigmoid(h_h)
        y_h = np.dot(U, h) + c
        y = sigmoid(y_h)

        #
        # Back Propagation
        #
        loss += np.sum((y - Y[i]) ** 2 / 2)
        dL = (y - Y[i])

        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        dU = np.zeros_like(U)
        dc = np.zeros_like(c)

        db = np.dot(U.T, dL * deriv_sigmoid(y_h)) * deriv_sigmoid(h_h)
        dW = np.dot(db, X[i].T)
        dc = dL * deriv_sigmoid(y_h)
        dU = np.dot(dc, h.T)

        # Updating Weights and Biases
        c -= lr * dc
        U -= lr * dU
        b -= lr * db
        W -= lr * dW

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
    x = X[i].reshape(2, 1)
    h = sigmoid(np.dot(W, x) + b)
    y = sigmoid(np.dot(U, h) + c)
    print(" {} XOR  {} => {:.4f}".format(X[i][0][0], X[i][1][0], y[0][0]))

print("========================")


#
#
#
hm = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        _x0, _x1 = float(i) / 100, float(j) / 100

        h_h = np.dot(W, [[_x0], [_x1]]) + b
        h = sigmoid(h_h)
        y_h = np.dot(U, h) + c
        y = sigmoid(y_h)

        hm[i][j] = y.item()

plt.title("")
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(False)
plt.ylim([-10, 110])
plt.xlim([-10, 110])
plt.plot(0, 0, "*", markersize=10, color="b")
plt.plot(100, 0, "*", markersize=10, color="r")
plt.plot(0, 100, "*", markersize=10, color="r")
plt.plot(100, 100, "*", markersize=10, color="b")

plt.imshow(hm)
plt.xticks([0, 50, 100], ["0", "0.5", "1.0"])
plt.yticks([0, 50, 100], ["0", "0.5", "1.0"])

cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)

plt.show()
