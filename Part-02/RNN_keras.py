#
# Sine wave prediction using Simple-RNN of Keras.
#
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
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import DataSet as ds


def plot_fig(model, wave_data, n_sample, n_sequence):

    wave = wave_data
    z = wave[0:n_sequence]
    input = wave[0:n_sequence+1]
    sin = [None for i in range(n_sequence)]
    gen = [None for i in range(n_sequence)]

    for j in range(n_sample):
        y = model.predict(z.reshape(1, n_sequence, 1))
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

lr = 0.001

input_units = 1
hidden_units = 32
output_units = 1

model = Sequential()
model.add(
    SimpleRNN(
        hidden_units,
        activation="tanh",
        use_bias=True,
        input_shape=(n_sequence, input_units),
        return_sequences=False,
    )
)
model.add(Dense(output_units, activation="linear", use_bias=True))
model.compile(loss="mean_squared_error", optimizer=Adam(lr))

print(model.summary())


# ============================
# Training
# ============================

n_epochs = 50

batch_size = 5

history_rst = model.fit(
    X,
    Y,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
)


plt.plot(history_rst.history["loss"], label="loss")
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
