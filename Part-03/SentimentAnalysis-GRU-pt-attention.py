#
# Sentiment Analysis using RNN(GRU) with Attention
#
# Developed environment:
# Python                       3.11.x
# torch                        2.x (MPS support: macOS Apple Silicon / CPU fallback)
# numpy                        1.26.x
# matplotlib                   3.9.x
# scikit-learn                 1.5.x
#
#   Copyright (c) 2024-2026, Hironobu Suzuki @ interdb.jp

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
import Attention_pt as Attention  # PyTorch version of Attention.py
import sys

# ========================================
# Device configuration (MPS / CPU)
# ========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# input_text = "imdb_labelled.txt"
input_text = "yelp_labelled.txt"
# input_text = "amazon_cells_labelled.txt"

# ========================================
# Create dataset
# ========================================
split = True

file_path = "../DataSets/sentiment_labelled_sentences/" + input_text

Lang = tokenizer.Tokenizer(file_path, category=True)
Lang.read_dataset(padding=True)

vocab_size = Lang.vocab_size()
max_len = Lang.max_len()

x = []
y = []

_pad = np.full(max_len, "<pad>")
_pad = np.array(_pad)


def padding(seq):
    w_len = len(seq)
    idxs = map(lambda w: Lang.word2idx[w], _pad[w_len:])
    for z in idxs:
        seq.append(z)
    return seq


for _, (sentence, label) in enumerate(zip(Lang.sentences, Lang.labels)):
    x.append(padding(sentence))
    y.append([label])

if split == True:
    X = np.array(x)
    Y = np.array(y)
    (X_train, X_validate, Y_train, Y_validate) = train_test_split(X, Y, test_size=0.2)
else:
    X_train = np.array(x)
    Y_train = np.array(y)

BUFFER_SIZE = int(X_train.shape[0])
BATCH_SIZE = 32
N_BATCH = BUFFER_SIZE // BATCH_SIZE


# Equivalent to tf.data.Dataset.from_tensor_slices(...).shuffle(BUFFER_SIZE).batch()
# Shuffle every epoch (same as TF version)
def make_batches(X, Y, batch_size):
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    n_batch = len(X) // batch_size
    batches = []
    for i in range(n_batch):
        s = i * batch_size
        e = s + batch_size
        batches.append((X[s:e], Y[s:e]))
    return batches


# ========================================
# Create Model
# ========================================

input_nodes = 1
hidden_nodes = 128
output_nodes = 1

embedding_dim = 64


class SentimentAnalysis(nn.Module):
    def __init__(self, hidden_units, output_units, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Many-to-Many GRU:
        #   return_sequences=True, return_state=True equivalent.
        #   nn.GRU returns (output, h_n) where:
        #     output: (batch, seq_len, hidden_units) <- all hidden states (= values for attention)
        #     h_n   : (1, batch, hidden_units)       <- final hidden state (= query for attention)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_units,
            batch_first=True,
        )
        # kernel_initializer="glorot_normal" equivalent
        nn.init.xavier_normal_(self.gru.weight_ih_l0)
        # recurrent_initializer="orthogonal" equivalent
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        # Bahdanau Attention:
        #   query  = state  (final hidden state, Fig.14-5)
        #   values = output (all hidden states,  Fig.14-5)
        self.attention = Attention.BahdanauAttention(hidden_nodes)

        # Dense(output_units, activation="sigmoid") equivalent.
        # sigmoid + MSELoss: no double-application issue (MSELoss has no internal activation).
        self.dense = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # return_sequences=True, return_state=True equivalent:
        #   output: (batch, seq_len, hidden_units) <- all hidden states
        #   h_n   : (1, batch, hidden_units)       <- final hidden state
        output, h_n = self.gru(x)

        # state: (batch, hidden_units) <- final hidden state = attention query
        # equivalent to TF: output, state = self.gru(x)
        state = h_n.squeeze(0)  # (batch, hidden_units)

        # Bahdanau attention:
        #   query  = state  (final hidden state)
        #   values = output (all hidden states)
        # equivalent to TF: context_vector, _ = self.attention(state, output)
        context_vector, attention_weights = self.attention(state, output)

        # sigmoid output: (batch, output_units)  in range [0, 1]
        x = torch.sigmoid(self.dense(context_vector))

        # Return both prediction and attention_weights,
        # equivalent to TF: return x, attention_weights
        return x, attention_weights


model = SentimentAnalysis(hidden_nodes, output_nodes, vocab_size, embedding_dim)
model.to(device)
print(model)

# ========================================
# Training
# ========================================

lr = 0.001
beta1 = 0.99
beta2 = 0.9999

# MeanSquaredError equivalent.
# sigmoid output is passed directly — no double-application issue.
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


def train(x, y):
    # numpy -> tensor
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    model.train()
    optimizer.zero_grad()
    # forward() returns (pred, attention_weights); only pred is used for loss
    pred, _ = model(x)  # pred: (batch, 1)
    loss = loss_function(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def comp(y, Y):
    if np.abs(y - Y) > 1 / 2:
        return False
    return True


def cal_accuracy():
    if split == True:
        model.eval()
        num_correct = 0
        with torch.no_grad():
            for _, (X, Y) in enumerate(zip(X_validate, Y_validate)):
                x = X[np.newaxis, ...]
                x = torch.tensor(x, dtype=torch.long).to(device)
                # Forward Propagation
                y = model(x)
                y = y[0].cpu().numpy()
                if comp(y, Y) == True:
                    num_correct += 1
        return num_correct / len(X_validate)
    else:
        return None


if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 40

history_loss = []
# train_loss accumulates like TF's metrics.Mean():
# it keeps a running total across epochs (not reset each epoch)
_train_loss_total = 0.0
_train_loss_count = 0

for epoch in range(1, n_epochs + 1):
    _loss = 0.0
    batches = make_batches(X_train, Y_train, BATCH_SIZE)

    for X_batch, Y_batch in batches:
        loss = train(X_batch, Y_batch)
        _loss += loss

    # equivalent to: train_loss(_loss); history_loss.append(train_loss.result())
    # TF's metrics.Mean() accumulates all values since reset and returns their mean.
    # Since it is never reset here, result() is the running mean over all epochs.
    _train_loss_total += _loss
    _train_loss_count += 1
    train_loss_result = _train_loss_total / _train_loss_count
    history_loss.append(train_loss_result)

    if epoch % 10 == 0 or epoch == 1:
        accuracy = cal_accuracy()
        if accuracy is None:
            print(
                "epoch: {}/{}, loss: {:.3}".format(epoch, n_epochs, train_loss_result)
            )
        else:
            print(
                "epoch: {}/{}, loss: {:.3}  accuracy: {:>5f}".format(
                    epoch, n_epochs, train_loss_result, accuracy
                )
            )

if n_epochs > 0:
    plt.plot(history_loss, label="loss")
    plt.legend(loc="best")
    plt.title("Training Loss History")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()

# ========================================
# Test
# ========================================

num_candidate = 20


def i2w(i):
    return "Positive" if i == 1 else "Negative"


if split == True:
    keys = np.arange(len(X_validate))
else:
    keys = np.arange(Lang.num_texts())
    X_validate = np.array(x)
    Y_validate = np.array(y)

n = np.minimum(num_candidate, len(keys))
keys = np.random.permutation(keys)[:n]

model.eval()
_correct = 0

for key in keys:
    x_seq = X_validate[key]

    sentence = []
    print("Text:", end="")
    for w in x_seq:
        if Lang.idx2word[w] == "<sos>" or Lang.idx2word[w] == "<eos>":
            continue
        if Lang.idx2word[w] == "<pad>":
            break
        print(Lang.idx2word[w], end=" ")
        sentence.append(Lang.idx2word[w])
    print(".")

    x_t = x_seq[np.newaxis, ...]
    x_t = torch.tensor(x_t, dtype=torch.long).to(device)

    # Forward Propagation
    # equivalent to TF: y = model(x, training=False); y = y[0].numpy()
    with torch.no_grad():
        y, _ = model(x_t)
    y = y[0].cpu().numpy()

    print("Correct value   => ", i2w(Y_validate[key]))
    print("Estimated value => ", i2w(np.round(y)))
    if comp(y, Y_validate[key]) == False:
        print("*** Wrong ***")
    else:
        _correct += 1

    print("")

print("\n{} out of {} sentences are correct.".format(_correct, n))

if split == True:
    accuracy = cal_accuracy()
    print("Accuracy: {:>5f}".format(accuracy))
