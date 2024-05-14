#
# Virtual Dice
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import matplotlib.pyplot as plt
import numpy as np

"""
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)
"""


class Dice:
    def __init__(self, num=6):
        assert num > 0
        self.num = num

    def toss(self):
        return np.random.randint(1, self.num + 1)

    def get_sequence(self, sequence_size):
        assert sequence_size > 0
        seq = np.zeros(sequence_size, dtype=int)
        for i in range(sequence_size):
            seq[i] = self.toss()
        return seq


# ========================================
# Create dataset
# ========================================
sequence_size = 30000
dice = Dice()

seq = dice.get_sequence(sequence_size)

# ========================================
# Sampling
# ========================================

count = np.zeros(dice.num, dtype=int)

for i in range(len(seq)):
    count[seq[i] - 1] += 1

if len(seq) > 0:
    prob = count / sequence_size

# ========================================
# Show the probability
# ========================================

print("========================================")
print("P(Event) = |event|/|trial| = Probability")
print("----------------------------------------")
for i in range(1, len(prob) + 1):
    print(
        " P({})    => {:>5d} /{:>6d}  = {:>1.5f}".format(
            i, count[i - 1], sequence_size, prob[i - 1]
        )
    )
print("----------------------------------------")


x = np.arange(1, len(prob) + 1)
plt.bar(x, prob, tick_label=x, align="center")
#plt.ylim(0, 1)
plt.ylabel("prob.")

plt.show()
