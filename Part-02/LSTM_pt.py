#
# Sine wave prediction using LSTM of PyTorch.
#
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
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizers
from torchinfo import summary
import DataSet as ds


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


class LSTM(nn.Module):
    def __init__(self, input_units, hidden_units, output_units):
        super().__init__()
        self.lstm = nn.LSTM(input_units, hidden_units, batch_first=True)
        self.dense = nn.Linear(hidden_units, output_units)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        h, _ = self.lstm(x)
        y = self.dense(h[:, -1])
        return y


def plot_fig(model, wave_data, n_sample, n_sequence):

    model.eval()
    wave = wave_data
    z = wave[0:n_sequence]
    input = wave[0:n_sequence+1]
    sin = [None for i in range(n_sequence)]
    gen = [None for i in range(n_sequence)]

    for j in range(n_sample):
        _z = torch.Tensor(z).to(device)
        y = model(_z.reshape(1, n_sequence, 1)).data.cpu().numpy()
        z = np.append(z, y)[1:]
        z = z.reshape(-1, n_sequence, 1)
        gen.append(y[0, 0])
        sin.append(wave[j+n_sequence])

    plt.plot(input, color="b", label="input")
    plt.plot(sin, "--", color="#888888", label="sine wave")
    plt.plot(gen, color="r", label="predict")
    plt.title("Prediction")
    plt.legend()
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.show()


# ============================
# Data creation
# ============================
n_sequence = 25
n_data = 100

n_sample = n_data - n_sequence  # number of sample

sin_data = ds.create_wave(n_data, 0.05)
X, Y = ds.dataset(sin_data, n_sequence, False)

X = X.reshape(X.shape[0], X.shape[1], 1)
Y = Y.reshape(Y.shape[0], Y.shape[1])

# ============================
# Model creation
# ============================

input_units = 1
hidden_units = 32
output_units = 1

model = LSTM(input_units, hidden_units, output_units).to(device)

summary(model=LSTM(input_units, hidden_units, output_units), input_size=X.shape)


# ============================
# Training
# ============================

lr = 0.001
beta1 = 0.9
beta2 = 0.999

criterion = nn.MSELoss(reduction="mean")
optimizer = optimizers.Adam(
    model.parameters(), lr=lr, betas=(beta1, beta2), amsgrad=True
)

n_epochs = 300
history_loss = []

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0

    x = torch.Tensor(X).to(device)
    y = torch.Tensor(Y).to(device)
    model.train()
    preds = model(x)
    loss = criterion(y, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

    history_loss.append(train_loss)

    if epoch % 10 == 0:
        print("epoch: {}/{}, loss: {:.3}".format(epoch, n_epochs, train_loss))

#
#
#
plt.plot(history_loss, label="loss")
plt.legend(loc="best")
plt.title("Training Loss History")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid(True)
plt.show()


# ============================
#
# ============================

sin_data = ds.create_wave(n_data, 0.0)
plot_fig(model, sin_data, n_sample, n_sequence)
