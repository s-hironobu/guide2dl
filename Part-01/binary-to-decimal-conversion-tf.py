#
# A Neural Network with Softmax layer for learning binary to decimal conversion.
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

import keras
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
# tf.random.set_seed(0)

# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([[0], [1], [2], [3]])


# Convert into One-Hot vector
Y = keras.utils.to_categorical(Y, 4)

"""
Y = [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
"""

# ========================================
# Create model
# ========================================

input_nodes = 2
hidden_nodes = 3
output_nodes = 4


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
            activation="softmax",
        )

    # Forward Propagation
    def call(self, x, training=None):
        h = self.f(x)
        y = self.g(h)
        return y


model = SimpleNN(input_nodes, hidden_nodes, output_nodes)

loss_func = losses.CategoricalCrossentropy()
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
        pred = model(X)

        loss = loss_func(Y, pred)

        # Back Propagation
        grad = tape.gradient(loss, model.trainable_variables)

    # Weights and bias Update
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return loss


n_epochs = 800

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

print("-------------------------------------------")
print("(x0 x1) => prob(0) prob(1) prob(2) prob(3)")
print("===========================================")

for i in range(0, len(Y)):
    _x0 = X[i][0]
    _x1 = X[i][1]
    print(
        "({}   {}) => {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(
            _x0, _x1, result[i][0], result[i][1], result[i][2], result[i][3]
        )
    )

print("===========================================")

x = np.arange(output_nodes)
plt.subplots_adjust(wspace=0.4, hspace=0.8)

for i in range(4):
    plt.subplot(4, 1, i + 1)
    x_i = X[i]
    title = str(x_i[0]) + " " + str(x_i[1])
    plt.title(title)
    plt.bar(x, result[i], tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")

plt.show()
