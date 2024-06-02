import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return (1 - np.tanh(x)**2)



Range = 10
Div = 10

#
# Sigmoid
#
X = []
func = []
d1_func = []

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    func.append(sigmoid(i))
    d1_func.append(deriv_sigmoid(i))

plt.plot(X, func, color="b", label="f(x) = sigmoid(x)")
plt.plot(X, d1_func, color="r", label="df(x)/dx")

plt.title("sigmoid function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

#
# Tanh
#
X = []
func = []
d1_func = []

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    func.append(tanh(i))
    d1_func.append(deriv_tanh(i))

plt.plot(X, func, color="b", label="f(x) = tanh(x)")
plt.plot(X, d1_func, color="r", label="df(x)/dx")

plt.title("tanh function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
