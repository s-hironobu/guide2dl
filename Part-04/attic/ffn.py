import numpy as np
import tensorflow as tf

d_model = 8
d_ffn = 6


def relu(x):
    return np.maximum(0, x)


def ffn(d_model, d_ffn):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_ffn, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ]
    )


class FFN:
    def __init__(self):
        super(FFN, self).__init__()

        self.w1 = tf.keras.layers.Dense(d_ffn, activation="relu", use_bias=False)
        self.w2 = tf.keras.layers.Dense(d_model, use_bias=False)
        self.ffn = tf.keras.Sequential([self.w1, self.w2])

    def call(self, x):
        y = self.ffn(x)
        return y, self.w1.weights, self.w2.weights


n = np.arange(20).reshape(4, 5)
N = tf.constant(n)

ffn = FFN()

X, w1, w2 = ffn.call(N)

print("Input matrix:n\n", n)

x_tf = X.numpy()

print("=== FFN Result: ===")

print("FFN(n)=\n", x_tf)

print("=== Manual Calculation Result: ===")
m = []
for i in range(len(n)):
    a = np.dot(relu(np.dot(n[i], w1)), w2)
    print(
        "np.dot(relu(np.dot(n[{}], w1)), w2) = M[{}] = {}".format(
            str(i), str(i), str(a[0][0])
        )
    )
    m.append(a[0][0].tolist())
print("=== Comparison of Results: ===")

if np.allclose(m, x_tf):
    print("FFN == M")
else:
    print("FFN != N  --> NG!!!")
