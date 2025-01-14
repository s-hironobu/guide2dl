#
# Single Hidden Layer Neural Network for learning XOR gate
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

from keras.models import Sequential
try:
    from keras.layers.core import Activation, Dense
except:
    from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from Common.ActivationFunctions import sigmoid


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

# ========================================
# Create model
# ========================================
input_nodes = 2
hidden_nodes = 3
output_nodes = 1


model = Sequential()
model.add(Dense(hidden_nodes, input_dim=input_nodes))
model.add(Activation("sigmoid"))
model.add(Dense(output_nodes))
model.add(Activation("sigmoid"))

optimizer = Adam(learning_rate=0.1)

model.compile(loss="mean_squared_error", optimizer=optimizer)

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

print("------------------------")
print("x0 XOR x1 => result")
print("========================")

for i in range(0, len(Y)):
    _x0 = X[i][0][0]
    _x1 = X[i][1][0]
    print(" {} XOR  {} => {:.4f}".format(_x0, _x1, result[i][0]))
print("========================")


[W, b, U, c] = Layer.get_weights(model)

hm = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        x = [float(i) / 100, float(j) / 100]
        _h = sigmoid(np.dot(W.T, x) + b)
        hm[i][j] = sigmoid(np.dot(U.T, _h) + c).item()

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
