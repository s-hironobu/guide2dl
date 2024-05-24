#
# Basic Layer collection: Dense (aka Fully Connection), Conv2D, MaxPolling2D, Flatten.
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
import keras

#
# Dense Connection Layer, a.k.a. fully-connected Layer
#
class Dense:
    def __init__(self, input_size, output_size, activate_func=None, deriv_activate_func=None, activate_class=None):

        self.W = np.random.uniform(size=(output_size, input_size))
        self.b = np.random.uniform(size=(output_size, 1))

        self.activate_func = activate_func
        self.deriv_activate_func = deriv_activate_func
        self.activate_class = activate_class

    def get_grads(self):
        return [self.dW, self.db]

    def get_params(self):
        return [self.W, self.b]

    def num_params(self):
        return self.W.size + self.b.size

    def forward_prop(self, x):
        self.x = x
        self.h = np.dot(self.W, self.x) + self.b
        if self.activate_class == None:
            self.y = self.activate_func(self.h)
        else:
            self.y = self.activate_class.activate_func(self.h)
        return self.y

    def back_prop(self, grads):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        if self.activate_class == None:
            self.db = self.deriv_activate_func(self.h) * grads
        else:
            self.db = self.activate_class.deriv_activate_func(self.h, grads)
        self.dW = np.dot(self.db, self.x.T)

        return np.dot(self.W.T, self.db)


#
# 2D Convolution Layer.
#
# No padding, fixed stride (1).
#
class Conv2D:
    def __init__(self, n_kernels, kernel_size, n_channels):
        self.n_kernels = n_kernels  # number of output layers
        self.n_channels = n_channels  # number of input layers
        self.k_size = kernel_size
        self.kernels = np.random.randn(self.n_channels, self.n_kernels, self.k_size, self.k_size) / (self.k_size**2)

    def get_kernels(self):
        return self.kernels

    def get_grads(self):
        return [self.dk]

    def get_params(self):
        return [self.kernels]

    def num_params(self):
        return self.kernels.size + 0  # non-bias

    def forward_prop(self, image):
        self.last_image = image
        h, w, _ = image.shape
        output = np.zeros((h - self.k_size + 1, w - self.k_size + 1, self.n_channels, self.n_kernels))

        for c in range(self.n_channels):
            for i in range(h - self.k_size - 1):
                for j in range(w - self.k_size - 1):
                    im_region = image[i : (i + self.k_size), j : (j + self.k_size), c]
                    output[i, j, c, :] = np.sum(im_region * self.kernels[c, :, :, :], axis=(1, 2))

        output = np.sum(output, axis=2)
        return output

    def back_prop(self, dL, return_grads=False):
        self.dk = np.zeros_like(self.kernels)
        h, w, _ = self.last_image.shape

        for c in range(self.n_channels):
            for i in range(h - self.k_size - 1):
                for j in range(w - self.k_size - 1):
                    im_region = self.last_image[i : (i + self.k_size), j : (j + self.k_size), c]
                    for k in range(self.n_kernels):
                        self.dk[c, k, :, :] += dL[i, j, k] * im_region

        #
        if return_grads == True:
            dI = np.zeros_like(self.last_image)
            dw, dh, dch = dL.shape
            p_w = (w - dw) // 2
            p_h = (h - dh) // 2
            for k in range(self.n_kernels):
                deltaP = dL[:, :, k]
                deltaP = np.pad(deltaP, [(p_w,), (p_h,)], "constant")
                for i in range(h):
                    for j in range(w):
                        for s in range(self.k_size):
                            for t in range(self.k_size):
                                if 0 <= (i - s) and 0 <= (j - t):
                                    dI[i, j, :] += (deltaP[i - s, j - t] * self.kernels[:, k, s, t])

            return dI
        else:
            return None


#
# Max Pooling 2D layer.
#
class MaxPooling2D:
    def __init__(self, pool_size):
        self.p_size = pool_size

    def forward_prop(self, image):
        self.last_image = image
        h, w, n_kernels = image.shape
        new_h, new_w = h // self.p_size, w // self.p_size
        self.output = np.zeros((new_h, new_w, n_kernels))

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[
                    (self.p_size * i) : (self.p_size * i + self.p_size),
                    (self.p_size * j) : (self.p_size * j + self.p_size),
                    :,
                ]
                self.output[i, j, :] = np.amax(im_region, axis=(0, 1))

        return self.output

    def back_prop(self, dE):
        ret = np.zeros_like(self.last_image)
        h, w, ch = self.last_image.shape
        new_h, new_w = h // self.p_size, w // self.p_size

        for i in range(new_h):
            for j in range(new_w):
                for s in range(self.p_size):
                    for t in range(self.p_size):
                        for k in range(ch):
                            x = self.p_size * i + s
                            y = self.p_size * j + t
                            if self.last_image[x, y, k] == self.output[i, j, k]:
                                ret[x, y, k] = dE[i, j, k]
                                break
        return ret


#
# Flatten layer.
#
class Flatten:
    def __init__(self, output_dim=None):
        self.output_dim = output_dim

    def flatten(self, image):
        self.last_image_shape = image.shape
        ret = image.flatten()
        if ret.ndim == 1 and self.output_dim == 2:
            ret = ret.reshape(ret.shape[0], 1)
        return ret

    def de_flatten(self, data):
        return data.reshape(self.last_image_shape)
