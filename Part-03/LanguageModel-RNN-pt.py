#
# RNN-based Language Model using GRU.
#
# Developed environment:
# Python                       3.11.x
# torch                        2.x (MPS support: macOS Apple Silicon / CPU fallback)
# numpy                        1.26.x
# matplotlib                   3.9.x
#
#   Copyright (c) 2024-2026, Hironobu Suzuki @ interdb.jp

import torch
import torch.nn as nn
from torch import optim
import logging
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
import sys

logging.basicConfig(level=logging.WARNING)

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

#
# Use/Write CHECKPOINT data or not.
#
CHECKPOINT = True

# input_text = 'eng-14.txt'
# input_text = 'eng-41.txt'
# input_text = 'eng-99.txt'
# input_text = "eng-150.txt"
# input_text = "eng-200.txt"
# input_text = "eng-250.txt"
input_text = "eng-300.txt"

# ========================================
# Create dataset
# ========================================

file_path = "../DataSets/small_vocabulary_sentences/" + input_text

#
# Read input text data
#
Lang = tokenizer.Tokenizer(file_path)
Lang.read_dataset(padding=True)

vocab_size = Lang.vocab_size()
max_len = Lang.max_len() + 1

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


#
# This model predicts the next word for an input word sequence x.
# Thus, y[i+1] = x[i].
#
# input : x = [<sos>, w[1], w[2],...w[n], <eos>, <pad>, ... , <pad>]
# output: y = [w[1], w[2],...w[n], <eos>, <pad>, <pad>, ... , <pad>]

for i in range(Lang.num_texts()):
    seq = Lang.sentences[i]
    x.append(padding(seq))
    y.append(padding(seq[1:]))

X_train = np.array(x)
Y_train = np.array(y)

BUFFER_SIZE = int(X_train.shape[0])
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE


# Equivalent to tf.data.Dataset.from_tensor_slices(...).batch()
def make_batches(X, Y, batch_size):
    n_batch = len(X) // batch_size
    batches = []
    for i in range(n_batch):
        s = i * batch_size
        e = s + batch_size
        batches.append((X[s:e], Y[s:e]))
    return batches


print("=================================")
print("vocabulary size: {}".format(vocab_size))
print("number of sentences: {}".format(Lang.num_texts()))
print("=================================")

# ========================================
# Create Model
# ========================================

input_nodes = 1
hidden_nodes = 1024
output_nodes = vocab_size

embedding_dim = 256


class GRU(nn.Module):
    def __init__(self, hidden_units, output_units, vocab_size, embedding_dim):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_units,
            batch_first=True,
        )
        # recurrent_initializer="glorot_uniform" equivalent
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        # Dense(output_units) — no activation here.
        # TF uses Dense(activation="softmax"), but nn.CrossEntropyLoss applies
        # log_softmax internally. Applying softmax here would cause double-softmax,
        # corrupting gradients. Softmax is applied only at prediction time.
        self.fc = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        output, _ = self.gru(x)  # (batch, seq_len, hidden_units)
        x = self.fc(output)  # (batch, seq_len, output_units) <- raw logits
        return x


model = GRU(hidden_nodes, output_nodes, vocab_size, embedding_dim)
model.to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ========================================
# Loss / Optimizer
# ========================================

lr = 0.0001
beta1 = 0.99
beta2 = 0.9999

# SparseCategoricalCrossentropy(reduction="none") equivalent.
# ignore_index=0 masks <pad> (index=0) automatically — same as TF mask logic.
loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))


def loss_function(real, pred):
    # real: (batch, seq_len)
    # pred: (batch, seq_len, vocab_size) <- raw logits
    #
    # CrossEntropyLoss expects (N, C, d1) so permute to (batch, vocab_size, seq_len)
    loss_ = loss_object(pred.permute(0, 2, 1), real)
    # ignore_index=0 already masks <pad>; just average remaining tokens
    return loss_.mean()


def compute_accuracy(pred, real):
    # pred: (batch, seq_len, vocab_size)  raw logits
    # real: (batch, seq_len)
    predicted_ids = pred.argmax(dim=-1)
    mask = real != 0
    correct = (predicted_ids == real) & mask
    return correct.sum().item() / mask.sum().item()


# ========================================
# Training
# ========================================

checkpoint_path = (
    "./checkpoints/rnn-language-model"
    + "-vocab-"
    + str(vocab_size)
    + "-embedding-"
    + str(embedding_dim)
    + "-hidden-"
    + str(hidden_nodes)
    + ".pt"
)

if CHECKPOINT and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Latest checkpoint restored!!")


def train(x, y):
    # numpy -> tensor
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    model.train()
    optimizer.zero_grad()
    output = model(x)  # (batch, seq_len, vocab_size)  raw logits
    loss = loss_function(y, output)
    acc = compute_accuracy(output, y)
    loss.backward()
    optimizer.step()
    return loss.item(), acc


# If n_epochs = 0, this model uses the trained parameters saved in the last checkpoint,
# allowing you to perform last word prediction without retraining.
if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 200

log_interval = 10

start = time.time()
for epoch in range(1, n_epochs + 1):

    total_loss = 0.0
    total_acc = 0.0
    batches = make_batches(X_train, Y_train, BATCH_SIZE)

    for X_batch, Y_batch in batches:
        loss, acc = train(X_batch, Y_batch)
        total_loss += loss
        total_acc += acc

    avg_loss = total_loss / BATCH_SIZE
    avg_acc = total_acc / N_BATCH

    if epoch % log_interval == 0 or epoch == 1:
        print(
            "epoch: {}/{}, loss: {:.3}  Accuracy: {:.4f}".format(
                epoch, n_epochs, avg_loss, avg_acc
            )
        )
        print(
            "Time taken for {} epoch {:.4f} sec\n".format(
                1 if epoch == 1 else log_interval, time.time() - start
            )
        )
        start = time.time()
        if CHECKPOINT:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print("Saving checkpoint for epoch {} at {}".format(epoch, checkpoint_path))


# ========================================
# the last word prediction
# ========================================

num_candidate = 5

keys = np.arange(Lang.num_texts())
X_validate = np.array(x)

n = np.minimum(num_candidate, len(keys))
keys = np.random.permutation(keys)[:n]


def get_len(sentence):
    for i, w in enumerate(sentence):
        if Lang.idx2word[w] == "<eos>":
            return i - 1


words = []
vals = []

model.eval()
with torch.no_grad():
    for i in keys:

        print("Text:  {}".format(Lang.texts[i]))

        x = X_validate[i]
        _len = get_len(x)

        x = x[:_len]  # Remove the last word and the words after it.
        print("Input: {}".format(str(Lang.detokenize(x))))
        x = x[np.newaxis, ...]

        x_t = torch.tensor(x, dtype=torch.long).to(device)
        output = model(x_t)  # (1, seq_len, vocab_size)  raw logits

        # Extract the final time step and apply softmax to get probabilities.
        # equivalent to: y = output[0]; y = y[-1].numpy()
        # TF's Dense(activation="softmax") already output probabilities,
        # so we apply softmax here at prediction time only.
        y = output[0]  # (seq_len, vocab_size)
        y = torch.softmax(y[-1], dim=-1).cpu().numpy()  # (vocab_size,)

        for _w in ["<sos>", "<eos>", "<pad>"]:
            y[Lang.word2idx[_w]] = 0.0

        w = []
        v = []

        for _ in range(5):
            _max_idx = np.argmax(y)
            _max_val = np.max(y)
            w.append(Lang.idx2word[_max_idx])
            v.append(_max_val)

            y[_max_idx] = 0.0

        words.append(w)
        vals.append(v)

        print("Predicted last word:")
        for j in range(len(w)):
            print("\t{}\t=> {:>5f}".format(w[j], v[j]))
        print("")


x = np.arange(len(keys))
plt.subplots_adjust(wspace=0.4, hspace=1.6)
for i in x:
    plt.subplot(num_candidate, 1, i + 1)
    plt.title(Lang.texts[keys[i]])
    plt.bar(x, vals[i], tick_label=words[i], align="center")
    plt.ylim(0, 1)
    plt.ylabel("prob.")

plt.show()
