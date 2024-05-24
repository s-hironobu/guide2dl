#
# CNN for MNIST using PyTorch.
#
# Developed environment:
# Python                       3.11.5
# keras                        2.15.0
# pip                          24.0
# numpy                        1.26.4
# matplotlib                   3.9.0
# scikit-learn                 1.5.0
# torch                        2.4.0.dev20240523
# torchaudio                   2.2.0.dev20240523
# torchinfo                    1.8.0
# torchvision                  0.19.0.dev20240523
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


# ========================================
# Load dataset from Keras repository.
# ========================================
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


batch_size = 128

# Create batches
num_X_train = len(X_train)
num_train_set = num_X_train // batch_size

X_train = X_train[0 : num_train_set * batch_size].reshape(
    num_train_set, batch_size, 1, 28, 28
)
X_train = torch.Tensor(X_train)

y_train = y_train[0 : num_train_set * batch_size].reshape(num_train_set, batch_size, 1)
y_train = y_train.reshape(num_train_set, batch_size).astype(np.int32)
y_train = torch.from_numpy(y_train)

output_size = np.unique(y_train).size

# ========================================
# Create Model
# ========================================

n_kernels = 8
kernel_size = 5
pool_size = 2


class Net(nn.Module):
    def __init__(self, image_size, kernel_size, n_kernels, pool_size, output_size):
        super(Net, self).__init__()
        self.conv2D = nn.Conv2d(1, n_kernels, kernel_size)
        self.conv2D_2 = nn.Conv2d(n_kernels, n_kernels, kernel_size)
        h1 = (image_size - kernel_size + 1) // pool_size
        h2 = (h1 - kernel_size + 1) // pool_size
        input_size = h2**2 * n_kernels
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.conv2D(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pool_size)
        x = F.relu(x)

        x = self.conv2D_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, pool_size)
        x = F.relu(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x


model = Net(28, kernel_size, n_kernels, pool_size, 10).to(device)

summary(
    model=Net(28, kernel_size, n_kernels, pool_size, 10), input_size=X_train[0].shape
)

# ========================================
# Training
# ========================================

model.train()  # Set training mode

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 5

for epoch in range(1, n_epochs + 1):
    loss_sum = 0

    for i in range(num_train_set):
        inputs, labels = X_train[i], y_train[i]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss_sum += loss

        # Backprop
        loss.backward()
        optimizer.step()

        if (i % 50 == 0 and i != 0) or i == 1:
            print(
                "Epoch {}/{}, Batch {} Loss: {:.5f}".format(
                    epoch, n_epochs, i, loss.item()
                )
            )

    print(
        "Epoch {}/{}, Loss: {:.5f}\n".format(
            epoch, n_epochs, loss_sum.item() / len(X_train)
        )
    )


# ========================================
# Test
# ========================================

model.eval()  # Set evaluation mode

x = np.arange(output_size)
r = np.random.randint(0, len(X_test), 5)

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i in range(5):
    k = r[i]
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(X_test[k, :].reshape(28, 28))

    plt.subplot(5, 2, 2 * i + 2)

    inputs = torch.Tensor(X_test[k, :].reshape(1, 1, 28, 28))
    inputs = inputs.to(device)

    res = model(inputs)
    res = res.to("cpu").detach().numpy().copy()
    plt.bar(x, res[0], tick_label=x, align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")
    plt.xlabel("candidate")

plt.show()
