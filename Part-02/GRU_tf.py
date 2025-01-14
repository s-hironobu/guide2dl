#
# Sine wave prediction using GRU of Tensorflow.
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
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, losses, metrics
import DataSet as ds


class GRU(tf.keras.Model):
    def __init__(self, hidden_units, output_units):
        super().__init__()
        self.gru = tf.keras.layers.GRU(
            hidden_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_initializer="glorot_normal",
            recurrent_initializer="orthogonal",
        )
        self.dense = tf.keras.layers.Dense(output_units, activation="linear")

    def call(self, x):
        x = self.gru(x)
        x = self.dense(x)
        return x

def plot_fig(model, wave_data, n_sample, n_sequence):

    wave = wave_data
    z = wave[0:n_sequence]
    input = wave[0:n_sequence+1]
    sin = [None for i in range(n_sequence)]
    gen = [None for i in range(n_sequence)]

    for j in range(n_sample):
        y = model.predict(z.reshape(1, z.shape[0], 1))
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
# Data creation
# ============================
n_sequence = 25
n_data = 100

n_sample = n_data - n_sequence  # number of sample

sin_data = ds.create_wave(n_data, 0.05)
X, Y = ds.dataset(sin_data, n_sequence, False)

X = X.reshape(X.shape[0], X.shape[1], 1)
Y = Y.reshape(Y.shape[0], Y.shape[1])

# ============================
# Model creation
# ============================

input_units = 1
hidden_units = 32
output_units = 1

model = GRU(hidden_units, output_units)

model.build(input_shape=(None, n_sequence, input_units))
model.summary()

# ============================
# Training
# ============================

lr = 0.001
beta1 = 0.9
beta2 = 0.999

criterion = losses.MeanSquaredError()
optimizer = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
train_loss = metrics.Mean()

n_epochs = 300
history_loss = []

for epoch in range(1, n_epochs + 1):

    with tf.GradientTape() as tape:
        preds = model(X)
        loss = criterion(Y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

    history_loss.append(train_loss.result())

    if epoch % 10 == 0:
        print("epoch: {}/{}, loss: {:.3}".format(epoch, n_epochs, train_loss.result()))

#
#
#
plt.plot(history_loss, label="loss")
plt.legend(loc="best")
plt.title("Training Loss History")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid(True)
plt.show()


# ============================
# Prediction
# ============================

sin_data = ds.create_wave(n_data, 0.0)
plot_fig(model, sin_data, n_sample, n_sequence)
