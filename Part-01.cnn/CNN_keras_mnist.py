#
# CNN for MNIST using Keras.
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#  keras                    2.10.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# ========================================
# Load dataset from Keras repository.
# ========================================

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Categorize labels
y_train_categorical = keras.utils.to_categorical(y_train)
y_test_categorical = keras.utils.to_categorical(y_test)


# ========================================
# Create Model
# ========================================

n_kernels = 8
kernel_size = 5
pool_size = 2

output_size = np.unique(y_train).size

batch_size = 128

validation_split = 0.2


def create_model():
    model = Sequential()
    model.add(
        Conv2D(
            n_kernels,
            kernel_size=kernel_size,
            use_bias=False,
            padding="valid",
            input_shape=X_train[0].shape,
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(
        Conv2D(
            n_kernels,
            kernel_size=kernel_size,
            use_bias=False,
            padding="valid",
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Flatten())
    model.add(Dense(output_size))
    model.add(Activation("softmax"))

    return model


model = create_model()
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

model.summary()

# ========================================
# Training
# ========================================

n_epochs = 5

model.fit(
    X_train,
    y_train_categorical,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_split=validation_split,
)

score = model.evaluate(X_test, y_test_categorical, verbose=0)

print("loss:", score[0])
print("accuracy:", score[1])

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
    res = model.predict(X_test[k].reshape(1, 28, 28, 1))
    plt.bar(x, res.reshape(10), tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")
    plt.xlabel("candidate")

plt.show()

#
# Show parameter lists
#
"""
for i in range(len(model.get_weights())):
    print(
        "model.weights[{}] = {}  shape => {}".format(
            i, model.weights[i].name, model.weights[i].shape
        )
    )
"""

#
# Show kernels
#
for K in range(2):
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
