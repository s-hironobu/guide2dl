#
# Display sparse transformer's strided and fixed attentions
#
# Generating Long Sequences with Sparse Transformers: https://arxiv.org/pdf/1904.10509.pdf

import math

N = 12  # max token size
s = 3  # Stride. Close to sqrt N
c = 1  # Hyperparameter


style = "UNICODE"
if style == "UNICODE":
    fil = "\u2B1B"
    emp = "\u2B1C"
else:
    fil = "1 "
    emp = "0 "


def strided(i, j):
    if ((i + s) > j and j > (i - s)) or (i - j) % s == 0:
        return True
    else:
        return False


def fixed(i, j):
    if (math.floor(j / s) == math.floor(i / s)) or ((j % s) >= (s - c)):
        return True
    else:
        return False


print("== Strided Attention ==")
for i in range(N):
    for j in range(N):
        if strided(i, j):
            print(fil, end="")
        else:
            print(emp, end="")
    print("")


print("== Fixed Attention ==")
for i in range(N):
    for j in range(N):
        if fixed(i, j):
            print(fil, end="")
        else:
            print(emp, end="")
    print("")

"""
print("== Strided & Fixed Attention ==")
for i in range(N):
    for j in range(N):
        if strided(i, j) or fixed(i, j):
            print(fil, end="")
        else:
            print(emp, end="")
    print("")
"""
