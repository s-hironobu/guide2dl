#
# RNN-based Language Model using GRU.
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

import keras
import logging
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
from tensorflow.keras import optimizers, losses, metrics
import sys

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings

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


dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

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


class GRU(tf.keras.Model):
    def __init__(self, hidden_units, output_units, vocab_size, embedding_dim):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.hidden_units,
            return_sequences=True,
            return_state=False,
            recurrent_initializer="glorot_uniform",
        )
        self.softmax = tf.keras.layers.Dense(output_units, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        output = self.gru(x)
        x = self.softmax(output)
        return x


model = GRU(hidden_nodes, output_nodes, vocab_size, embedding_dim)

model.build(input_shape=(None, max_len))
model.summary()

lr = 0.0001
beta1 = 0.99
beta2 = 0.9999

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
optimizer = optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # this masks '<pad>'
    """
    Example:

    real= tf.Tensor(
    [[21  1 44 0  0]   (jump !    <eos> <pad> <pad>)
    [ 17  9 24 2 44]   (i    go   there .     <eos>)
    [ 27  1 44 0  0]   (no   !    <eos> <pad> <pad>)
    [ 21 22 32 2 44]], (i    know you   .     <eos>)
    , shape=(4, 5), dtype=int64)

    where <pad> = 0.

    mask= tf.Tensor(
    [[True  True  True False False]
    [ True  True  True True  True ]
    [[True  True  True False False]
    [ True  True  True True  True ],
    shape=(4, 5), dtype=bool)
    """

    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

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
)


if CHECKPOINT == True:
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")


@tf.function
def train(x, y):
    with tf.GradientTape() as tape:
        output = model(x)
        loss = loss_function(y, output)
        train_accuracy(y, output)
        grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss


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
    train_accuracy.reset_states()

    for batch, (X_train, Y_train) in enumerate(dataset):
        loss = train(X_train, Y_train)
        total_loss += loss.numpy()

    avg_loss = total_loss / BATCH_SIZE

    if epoch % log_interval == 0 or epoch == 1:
        print(
            "epoch: {}/{}, loss: {:.3}  Accuracy: {:.4f}".format(
                epoch, n_epochs, avg_loss, train_accuracy.result()
            )
        )
        print(
            "Time taken for {} epoch {:.4f} sec\n".format(
                1 if epoch == 1 else log_interval, time.time() - start
            )
        )
        start = time.time()
        if CHECKPOINT == True:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch, ckpt_save_path))


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

for i in keys:

    print("Text:  {}".format(Lang.texts[i]))

    x = X_validate[i]
    _len = get_len(x)

    x = x[:_len]  # Remove the last word and the words after it.
    print("Input: {}".format(str(Lang.detokenize(x))))
    x = x[np.newaxis, ...]

    output = model(x)

    # Extract the final output.
    y = output[0]
    y = y[-1].numpy()

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
