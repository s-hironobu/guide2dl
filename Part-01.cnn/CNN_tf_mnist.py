#
# CNN for MNIST using Tensorflow.
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

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np


# ========================================
# Load dataset from Keras repository.
# ========================================
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255


batch_size = 128
n_X_train = len(X_train)
n_train_set = n_X_train // batch_size

X_train = X_train[
    0 : n_train_set * batch_size,
].reshape(n_train_set, batch_size, 28, 28, 1)
y_train = y_train[
    0 : n_train_set * batch_size,
].reshape(n_train_set, batch_size, 1)

X_train = tf.constant(X_train)
y_train = tf.constant(y_train)


# ========================================
# Create Model
# ========================================

n_kernels = 8
kernel_size = 5
pool_size = 2

output_size = np.unique(y_train).size


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(
            filters=n_kernels, kernel_size=kernel_size, activation="relu"
        )
        self.mp1 = MaxPool2D(
            pool_size=(pool_size, pool_size),
            strides=None,
            padding="valid",
            data_format=None,
        )

        self.conv2 = Conv2D(
            filters=n_kernels, kernel_size=kernel_size, activation="relu"
        )
        self.mp2 = MaxPool2D(
            pool_size=(pool_size, pool_size),
            strides=None,
            padding="valid",
            data_format=None,
        )

        self.flatten = Flatten()
        self.softmax = Dense(units=output_size, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        return self.softmax(x)


model = CNN()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


model.build(input_shape=(None, 28, 28, 1))
model.summary()

# ========================================
# Training
# ========================================


@tf.function
def train(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


n_epochs = 5

for epoch in range(1, n_epochs + 1):

    for i in range(n_train_set):
        train(X_train[i], y_train[i])

    print(
        "Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.5f}".format(
            epoch, n_epochs, train_loss.result(), train_accuracy.result()
        )
    )

# ========================================
# Test
# ========================================

#
# Show random selected 5 results.
#

x = np.arange(output_size)
r = np.random.randint(0, len(X_test), 5)

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i in range(5):
    k = r[i]
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(X_test[k, :].reshape(28, 28))

    plt.subplot(5, 2, 2 * i + 2)
    res = model.predict(tf.constant(X_test[k].reshape(1, 28, 28, 1)))
    plt.bar(x, res.reshape(10), tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")
    plt.xlabel("candidate")

plt.show()


#
# Show kernels
#
for K in [0, 2]:
    kw, kh, n_channels, n_kernels = model.weights[K].shape
    plt.subplots_adjust(wspace=0.4, hspace=0.6, left=0.05, right=0.75)

    for i in range(n_channels):
        for j in range(n_kernels):
            plt.subplot(8, 8, (i * n_kernels) + j + 1)
            kernel = model.weights[K]
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(kernel[:, :, i, j])

    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)

    plt.show()
