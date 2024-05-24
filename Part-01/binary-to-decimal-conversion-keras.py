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
from keras.models import Sequential
try:
    from keras.layers.core import Activation, Dense
except:
    from keras.layers import Activation, Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)

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

# Convert row vectors into column vectors.
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 4, 1)

# ========================================
# Create model
# ========================================
input_nodes = 2
hidden_nodes = 3
output_nodes = 4

model = Sequential()
model.add(Dense(hidden_nodes, input_dim=input_nodes))
model.add(Activation("sigmoid"))
model.add(Dense(output_nodes))
model.add(Activation("softmax"))

optimizer = Adam(learning_rate=0.1)

model.compile(loss="categorical_crossentropy", optimizer=optimizer)

print(model.summary())


# ========================================
# Training
# ========================================

n_epochs = 300

# Run through the data `epochs` times
history = model.fit(X, Y, epochs=n_epochs, batch_size=1, verbose=1)

#
# Show loss history
#
plt.plot(history.history["loss"], color="b", label="loss")
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
    _x0 = X[i][0][0]
    _x1 = X[i][1][0]
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
    title = str(x_i[0][0]) + " " + str(x_i[1][0])
    plt.title(title)
    plt.bar(x, result[i], tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")

plt.show()
