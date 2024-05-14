#
# Single Hidden Layer Neural Network for learning XOR gate
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#  torch                    2.0.1
#  torchinfo                1.8.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


# ========================================
# Create datasets
# ========================================

# Inputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# The ground-truth labels
Y = np.array([[0], [1], [1], [0]])

# convert numpy array to tensor
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()


# ========================================
# Create Model
# ========================================

input_nodes = 2
hidden_nodes = 3
output_nodes = 1


class SimpleNN(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)

    # Forward Propagation
    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


model = SimpleNN(input_nodes, hidden_nodes, output_nodes).to(device)

summary(model=SimpleNN(input_nodes, hidden_nodes, output_nodes), input_size=X.shape)

# ========================================
# Training
# ========================================

n_epochs = 10000

# set training mode
model.train()

# set training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

history_loss = []

#
# Training loop
#
for epoch in range(1, n_epochs + 1):
    train_loss = 0.0

    optimizer.zero_grad()

    # forward propagation
    outputs = model(torch.Tensor(X).to(device))

    # compute loss
    loss = criterion(outputs, torch.Tensor(Y).to(device))

    # Weights and Bias Update
    loss.backward()
    optimizer.step()

    # save loss of this epoch
    train_loss += loss.item()
    history_loss.append(train_loss)

    if epoch % 100 == 0 or epoch == 1:
        print("epoch: {} / {}  Loss = {:.4f}".format(epoch, n_epochs, loss))


#
# Show loss history
#
plt.plot(history_loss, color="b", label="loss")
plt.title("Training Loss History")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ========================================
# Test
# ========================================

print("------------------------")
print(" x0 XOR x1 => result")
print("========================")

model.eval()

for i in range(0, len(Y)):
    x0 = X[i][0]
    x1 = X[i][1]
    with torch.no_grad():
        result = model(
            torch.from_numpy(np.array([[x0, x1]])).to(
                dtype=torch.float32, device=device
            )
        )
        result = result.to("cpu").detach().numpy().copy()

    print(" {}  XOR {}  => {:.4f}".format(int(x0), int(x1), float(result)))

print("========================")
