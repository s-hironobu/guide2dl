#
# Single Hidden Layer Neural Network for learning XOR gate
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#  tensorflow-macos         2.10.0
#  tensorflow-metal         0.6.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from Common.ActivationFunctions import sigmoid


# np.random.seed(0)
# tf.random.set_seed(0)

# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([[0], [1], [1], [0]])

# ========================================
# Create model
# ========================================
input_nodes = 2
hidden_nodes = 3
output_nodes = 1


class SimpleNN(tf.keras.Model):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(SimpleNN, self).__init__()

        # Hidden layer
        self.f = tf.keras.layers.Dense(
            units=hidden_nodes,
            input_dim=input_nodes,
            activation="sigmoid",
        )

        # Output layer
        self.g = tf.keras.layers.Dense(
            units=output_nodes,
            activation="sigmoid",
        )

    # Forward Propagation
    def call(self, x, training=None):
        h = self.f(x)
        y = self.g(h)
        return y


model = SimpleNN(input_nodes, hidden_nodes, output_nodes)

loss_func = losses.MeanSquaredError()
optimizer = optimizers.Adam(learning_rate=0.1)

model.build(input_shape=(None, 2))
model.summary()

# ========================================
# Training
# ========================================

#
# Training function
#
@tf.function
def train(X, Y):
    with tf.GradientTape() as tape:
        # Forward Propagation
        y = model(X)

        # Back Propagation
        loss = loss_func(Y, y)
        grad = tape.gradient(loss, model.trainable_variables)

    # Weights and Bias Update
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return loss


n_epochs = 3000

history_loss = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):
    _loss = 0.0

    loss = train(X, Y)

    _loss += loss.numpy()
    history_loss.append(_loss)

    if epoch % 100 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.6f}".format(epoch, n_epochs, _loss))


#
# Show loss history
#
plt.plot(history_loss, color="b", label="loss")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ========================================
# Test
# ========================================

result = model.predict(X)

print("------------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(0, len(Y)):
    _x0 = X[i][0]
    _x1 = X[i][1]
    print(" {} XOR  {} => {:.4f}".format(_x0, _x1, result[i][0]))
print("========================")

#
#
#

[W, b, U, c] = Layer.get_weights(model)

hm = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        x = [float(i) / 100, float(j) / 100]
        _h = sigmoid(np.dot(W.T, x) + b)
        hm[i][j] = sigmoid(np.dot(U.T, _h) + c)

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
