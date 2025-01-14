#
# CNN for MNIST from scratch.
#
# Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Softmax
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
import matplotlib.pyplot as plt
import time
import sys

sys.path.append("..")
from Common import Optimizer, Layers
from Common.Optimizer import update_weights
from Common.ActivationFunctions import Softmax

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""


def summary(im_width, im_hight, n_kernels, kernel_size, n_kernels_2, kernel_size_2, pool_size, output_size):
    print("_________________________________________________________________")
    print(" Layer (type)                     Output Shape         Param #")
    print("=================================================================")
    cv = conv.num_params()
    params = cv
    cv_w = im_width - (kernel_size - 1)
    cv_h = im_width - (kernel_size - 1)

    print(
        " conv2d (Conv2D)                 ({:>3},{:>3},{:>2})          {:>8}".format(
            cv_w, cv_h, n_kernels, cv
        )
    )
    mp_w = cv_w // pool_size
    mp_h = cv_h // pool_size
    print(
        " max_pooling2d (MaxPooling2D)    ({:>3},{:>3},{:>2})                 0".format(
            mp_w, mp_h, n_kernels
        )
    )

    cv2 = conv2.num_params()
    params += cv2
    cv2_w = mp_w - (kernel_size_2 - 1)
    cv2_h = mp_h - (kernel_size_2 - 1)
    print(
        " conv2d_1 (Conv2D)               ({:>3},{:>3},{:>2})          {:>8}".format(
            cv2_w, cv2_h, n_kernels_2, cv2
        )
    )

    mp2_w = cv2_w // pool_size
    mp2_h = cv2_h // pool_size
    print(
        " max_pooling2d_1 (MaxPooling2D)  (None,{:>3},{:>3},{:>2})            0".format(
            mp2_w, mp2_h, n_kernels_2
        )
    )

    sm_param = softmax.num_params()

    params += sm_param
    print(
        " dense (Softmax)                 (None, {:>3})           {:>8}".format(
            output_size, sm_param
        )
    )
    print("=================================================================")
    print("Total params: {}".format(params))


#
#
#

# Load dataset
import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_images, im_width, im_hight = X_train.shape

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)  # to One-Hot vector
y_train = y_train.reshape(y_train.shape[0], 10, 1)  # from row-vector to column-vector
y_test = keras.utils.to_categorical(y_test, 10)  # to One-Hot vector
y_test = y_test.reshape(len(y_test), 10, 1)  # from row-vector to column-vector

# ========================================
# Create Model
# ========================================


# Convolution Layer
n_kernels = 8  # Number of kernels
n_kernels2 = 6  # Number of kernels
kernel_size = 5  # Kernel size

# MAxPooling layer
pool_size = 2

# Softmax Layer
output_size = 10

sm_w = (im_width - (kernel_size - 1)) // pool_size
sm_h = (im_width - (kernel_size - 1)) // pool_size


# Create instances
conv = Layers.Conv2D(n_kernels, kernel_size, 1)  # 1x28x28 -> 8x24x24
pool = Layers.MaxPooling2D(pool_size)  # 8x24x24 -> 8x12x12

sm2_w = (sm_w - (kernel_size - 1)) // pool_size
sm2_h = (sm_h - (kernel_size - 1)) // pool_size

conv2 = Layers.Conv2D(n_kernels2, kernel_size, n_kernels)  # 8x12x12 -> 8x8x8
pool2 = Layers.MaxPooling2D(pool_size)  # 8x8x8 -> 8x4x4

flatten = Layers.Flatten(output_dim=2)

softmax = Layers.Dense(
    sm2_w * sm2_h * n_kernels2, output_size, activate_class=Softmax()
)


summary(im_width, im_hight, n_kernels, kernel_size, n_kernels2, kernel_size, pool_size, output_size)

# ========================================
# Training
# ========================================


lr = 0.001

beta1 = 0.9
beta2 = 0.999
optimizer = Optimizer.Adam(lr=lr, beta1=beta1, beta2=beta2)


def train(image, label, lr, optimizer=None, max_norm=None):

    # Forward Propagation
    out = conv.forward_prop(image / 255)
    out = pool.forward_prop(out)
    out = conv2.forward_prop(out)
    out = pool2.forward_prop(out)
    out = flatten.flatten(out)
    out = softmax.forward_prop(out)

    # Back Propagation
    loss = -np.sum(label * np.log(out + 1e-8))
    acc = 1 if (np.argmax(out) == np.argmax(label)) else 0

    dL = -label / (out + 1e-8)
    dx = softmax.back_prop(dL)
    dx = flatten.de_flatten(dx)
    dx = pool2.back_prop(dx)
    dx = conv2.back_prop(dx, return_grads=True)
    dx = pool.back_prop(dx)
    _ = conv.back_prop(dx)

    # Weights and Bias Update
    update_weights([softmax, conv2, conv], optimizer=optimizer)

    return loss, acc


n_epochs = 3
n_images = 3500

for epoch in range(1, n_epochs + 1):
    start = time.time()
    print("\nepoch:{}/{}".format(epoch, n_epochs))

    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(X_train, y_train)):
        if i > n_images:
            break

        if i % 100 == 0 and i != 0:
            print(
                "\r [Step {}/{}] Past 100 steps: Average Loss {:.3f} | Accuracy: {:.3f}".format(
                    i, n_images, loss / 100, num_correct / 100
                ),
                end="",
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label, lr, optimizer)
        loss += l
        num_correct += acc

    print("\nTime taken for 1 epoch {:.4f} sec".format(time.time() - start))

# ========================================
# Test
# ========================================

#
# Show random selected 5 results.
#
def predict(image):

    # Forward Propagation
    out = conv.forward_prop(image / 255)
    out = pool.forward_prop(out)
    out = conv2.forward_prop(out)
    out = pool2.forward_prop(out)
    out = flatten.flatten(out)
    return softmax.forward_prop(out)


x = np.arange(output_size)
r = np.random.randint(0, len(X_test), 5)

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i in range(5):
    k = r[i]
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(X_test[k, :].reshape(28, 28))

    plt.subplot(5, 2, 2 * i + 2)
    ret = predict(X_test[k])
    plt.bar(x, ret.reshape(ret.shape[0]), tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")
    plt.xlabel("candidate")

plt.show()
