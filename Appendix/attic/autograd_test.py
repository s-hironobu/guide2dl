#
#
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

from autograd import grad

# ========================================
#
# ========================================

x = 0.5

# Back Propagation
_grad = 12 * x**2 * (2*x**3 + 3)

print("grad =", _grad)

print("===================")

# ========================================
#
# ========================================

def z(x):
    y = 2 * x**3 + 3
    z = y**2
    return z

gradient_fun = grad(z)
_grad = gradient_fun(x)
print("grad =", _grad)
