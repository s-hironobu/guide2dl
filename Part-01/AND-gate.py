#
# Perceptron for learning AND gate
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

# np.random.seed(0)

# ========================================
# Activation Function
# ========================================


def activate_func(x):
    # step function
    return 1 if x > 0 else 0


# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([[0], [0], [0], [1]])
OPERATION = "AND"

"""
Y = np.array([[0], [1], [1], [1]])
OPERATION = "OR"

Y = np.array([[1], [1], [1], [0]])
OPERATION = "NAND"

Y = np.array([[0], [1], [1], [0]])
OPERATION = "XOR"
"""

# ========================================
# Initialize random weights and bias
# ========================================

W = np.random.uniform(size=(2))
b = np.random.uniform(size=(1))

# ========================================
# Training
# ========================================

n_epochs = 150  # Epoch
lr = 0.01  # Learning rate

history_loss = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):
    loss = 0.0

    for i in range(0, len(Y)):
        # Forward Propagation
        x0 = X[i][0]
        x1 = X[i][1]
        y_h = W[0] * x0 + W[1] * x1 + b[0]
        y = activate_func(y_h)

        # Updating Weights and Biases
        loss += (y - Y[i]) ** 2 / 2

        W[0] -= lr * (y - Y[i]) * x0
        W[1] -= lr * (y - Y[i]) * x1
        b[0] -= lr * (y - Y[i])

    history_loss.append(loss / len(Y))
    if epoch % 10 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, loss[0]))

#
# Show loss history
#
plt.plot(history_loss, color="b", label="loss")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# ========================================
# Test
# ========================================

print("------------------------")
print("x0 {} x1 => result".format(OPERATION))
print("========================")

for x in X:
    y = activate_func(W[0] * x[0] + W[1] * x[1] + b[0])
    print(" {} {}  {} =>      {}".format(x[0], OPERATION, x[1], int(y)))

print("========================")

#
# Show Decision Line (Separator)
#
slope = -W[0] / W[1]
bias = -b[0] / W[1]

fig, ax = plt.subplots(1, 1)
title = "Decision Line  (" + OPERATION + "-gate)"
plt.title(title)
plt.xlabel("X[0]")
plt.ylabel("X[1]")
plt.grid(False)
plt.ylim([-0.1, 1.4])
plt.xlim([-0.1, 1.4])

for i in range(len(X)):
    x0, x1 = X[i]
    y = Y[i]
    color = "b" if y[0] == 0 else "r"
    plt.plot(x0, x1, "*", markersize=10, color=color)

ax.axline((0, bias), slope=slope)

plt.show()
