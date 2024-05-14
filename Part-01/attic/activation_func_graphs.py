import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def deriv_tanh(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return x if x >= 0 else 0 + 0.023


def deriv_relu(x):
    return 1 if x >= 0 else 0


X = []
_sigmoid = []
_tanh = []
_relu = []
_deriv_relu = []

Range = 3
Div = 50

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    _sigmoid.append(sigmoid(i))
    _tanh.append(tanh(i))
    _relu.append(relu(i))


plt.plot(X, _sigmoid, color="b", label="f(x)=sigmoid(x)")
plt.plot(X, _tanh, color="r", label="f(x)=tanh(x)")
plt.plot(X, _relu, color="g", label="f(x)=ReLu(x)")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
plt.show()



#
# Sigmoid
#
Range = 10
Div = 10

X = []
func = []
d1_func = []

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    func.append(sigmoid(i))
    d1_func.append(deriv_sigmoid(i))

#
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
Range = 10
Div = 10

X = []
func = []
d1_func = []

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    func.append(tanh(i))
    d1_func.append(deriv_tanh(i))

#
plt.plot(X, func, color="b", label="f(x) = tanh(x)")
plt.plot(X, d1_func, color="r", label="df(x)/dx")

plt.title("tanh function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()


#
# ReLU
#




X = []
_sigmoid = []
_tanh = []
_relu = []
_deriv_relu = []

Range = 3
Div = 50

for J in range(-1 * Range * Div, Range * Div):
    i = float(J / Div)
    X.append(i)

    """
    _sigmoid.append(sigmoid(i))
    _tanh.append(tanh(i))
    _relu.append(relu(i))
    """
    _relu.append(relu(i))
    if i == 0.0:
        _deriv_relu.append(None)
    else:
        _deriv_relu.append(deriv_relu(i))
#
"""
plt.plot(X, _sigmoid, color="b", label="f(x)=sigmoid(x)")
plt.plot(X, _tanh, color="r", label="f(x)=tanh(x)")
plt.plot(X, _relu, color="g", label="f(x)=ReLu(x)")
"""

plt.plot(X, _relu, color="b", label="f(x) = relu(x)")
plt.plot(X, _deriv_relu, color="r", label="df(x)/dx")


plt.xlabel("x")
plt.ylabel("f(x)")
# plt.grid()
plt.legend()
plt.show()
