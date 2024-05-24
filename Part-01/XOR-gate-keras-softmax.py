#
# A Neural Network with Softmax layer for learning XOR gate.
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
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Convert row vectors into column vectors.
X = X.reshape(4, 2, 1)
Y = Y.reshape(4, 2, 1)

# ========================================
# Create model
# ========================================
input_nodes = 2
hidden_nodes = 3
output_nodes = 2

model = Sequential()
model.add(Dense(hidden_nodes, input_dim=input_nodes))
model.add(Activation("sigmoid"))
model.add(Dense(output_nodes))
model.add(Activation("softmax"))

optimizer = Adam(learning_rate=0.1)

# model.compile(loss="binary_crossentropy", optimizer=optimizer)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

print(model.summary())


# ========================================
# Training
# ========================================

n_epochs = 500

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

print("-------------------------------------")
print("x0 XOR x1 => prob(0)   prob(1)")
print("=====================================")
for i in range(0, len(Y)):
    _x0 = X[i][0][0]
    _x1 = X[i][1][0]
    print(
        " {} XOR  {} =>  {:.4f}    {:.4f}".format(_x0, _x1, result[i][0], result[i][1])
    )
print("=====================================")


x = np.arange(2)
plt.subplots_adjust(wspace=0.4, hspace=0.8)

for i in range(4):
    plt.subplot(4, 1, i + 1)
    title = str(X[i][0][0]) + "  XOR  " + str(X[i][1][0])
    plt.title(title)
    plt.bar(x, result[i], tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")

plt.show()
