#
# Support Vector Machine for learning XOR gate
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

from sklearn import svm
from sklearn.metrics import accuracy_score

# ========================================
# Create datasets
# ========================================

# Inputs
X = [[0, 0], [1, 0], [0, 1], [1, 1]]

# The ground-truth labels
Y = [0, 1, 1, 0]

# ========================================
# Create model
# ========================================
model = svm.SVC()

# ========================================
# Training
# ========================================
model.fit(X, Y)

# ========================================
# Test
# ========================================
result = model.predict(X)

print("---------------------")
print("x0 XOR x1 => result")
print("=====================")

for i in range(len(X)):
    x = X[i]
    print(" {} XOR  {} =>   {}".format(x[0], x[1], int(result[i])))

print("=====================")
