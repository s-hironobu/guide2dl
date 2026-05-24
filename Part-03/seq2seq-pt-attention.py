#
# A translator from Spanish to English using seq2seq with Attention.
#
# This is modified based on https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ja
#
# Developed environment:
# Python                       3.11.x
# torch                        2.x (MPS support: macOS Apple Silicon / CPU fallback)
# numpy                        1.26.x
# matplotlib                   3.9.x
# scikit-learn                 1.5.x
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import Attention_pt as Attention
import logging
import time
import os
import sys

sys.path.append("..")
from Common import LanguageTranslationHelper as lth

logging.basicConfig(level=logging.WARNING)

#
# Use/Write CHECKPOINT data or not.
#
CHECKPOINT = True

#
# Select Attention Type.
#
AttentionType = "Bahdanau"  # a.k.a. Additive attention, Multi-Layer perceptron.
# AttentionType = "Luong"   # a.k.a. Bilinear, General attention.

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

# ========================================
# Define Classes
# ========================================


#
# Encoder
#
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        """
        Bahdanau attention model uses bidirectional RNN in the original paper,
        but we use a simple GRU.
        """
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=enc_units,
            batch_first=True,
        )
        # recurrent_initializer="glorot_uniform" equivalent
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

    def forward(self, x):
        x = self.embedding(x)

        # return_sequences=True, return_state=True equivalent:
        #   output: (batch, seq_len, enc_units)  <- all time steps
        #   h_n   : (1, batch, enc_units)        <- final hidden state
        output, h_n = self.gru(x)

        # state: (batch, enc_units)
        state = h_n.squeeze(0).contiguous()

        return output, state


#
# Decoder
#
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if AttentionType == "Bahdanau":
            self.attention = Attention.BahdanauAttention(dec_units)
            # GRU input = concat(context_vector, embedding)
            #   context_vector : (batch, 1, dec_units)
            #   embedding      : (batch, 1, embedding_dim)
            #   concat         : (batch, 1, dec_units + embedding_dim)
            self.gru = nn.GRU(
                input_size=dec_units + embedding_dim,
                hidden_size=dec_units,
                batch_first=True,
            )
        elif AttentionType == "Luong":
            self.attention = Attention.LuongAttention(dec_units)
            self.W = nn.Linear(2 * dec_units, 2 * dec_units)
            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=dec_units,
                batch_first=True,
            )
        else:
            print("Error: {} is not supported.".format(AttentionType))
            sys.exit()

        # recurrent_initializer="glorot_uniform" equivalent
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        out_dim = dec_units if AttentionType == "Bahdanau" else 2 * dec_units
        self.softmax = nn.Linear(out_dim, vocab_size)

    def forward(self, x, hidden, enc_output):
        # x         : (batch, 1)                  <- one token at a time
        # hidden    : (batch, dec_units)           <- attention query (updated each step)
        # enc_output: (batch, src_seq_len, enc_units)

        x = self.embedding(x)  # (batch, 1, embedding_dim)

        if AttentionType == "Bahdanau":
            # Use hidden as attention query (same as TF version)
            context_vector, attention_weights = self.attention(hidden, enc_output)

            # Concat context_vector and embedding
            # equivalent to: x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            context_vector_exp = context_vector.unsqueeze(1)  # (batch, 1, dec_units)
            x = torch.cat(
                [context_vector_exp, x], dim=-1
            )  # (batch, 1, dec_units+embedding_dim)

            # GRU with NO initial_state (zeros), faithful to TF version.
            # TF: output, state = self.gru(inputs=x)   <- no initial_state
            # The returned state becomes the attention query for the NEXT step.
            output, h_n = self.gru(x)

            # output: (batch, 1, dec_units) -> (batch, dec_units)
            # equivalent to: output = tf.reshape(output, (-1, output.shape[2]))
            output = output.squeeze(1)

        else:
            # Luong: run GRU with initial_state=hidden first, then compute attention
            # equivalent to: output, state = self.gru(inputs=x, initial_state=hidden)
            hidden_h0 = hidden.unsqueeze(0).contiguous()
            output, h_n = self.gru(x, hidden_h0)
            output = output.squeeze(1)

            state_for_attn = h_n.squeeze(0).contiguous()
            context_vector, attention_weights = self.attention(
                state_for_attn, enc_output
            )

            # equivalent to: output = tf.nn.tanh(self.W(tf.concat([state, context_vector], axis=-1)))
            output = torch.tanh(
                self.W(torch.cat([state_for_attn, context_vector], dim=-1))
            )

        # state: returned as attention query for next step (same role as TF's `state`)
        state = h_n.squeeze(0).contiguous()  # (batch, dec_units)

        # Output raw logits — do NOT apply softmax here.
        # TF uses Dense(activation="softmax"), but nn.CrossEntropyLoss already
        # applies log_softmax internally.  Applying softmax here would cause
        # double-softmax, which corrupts gradients and prevents learning.
        # Softmax is applied only in evaluate() for greedy search.
        x = self.softmax(output)  # (batch, vocab_size)  <- raw logits

        return x, state, attention_weights


# ========================================
# Create Dataset
# ========================================

num_examples = 110000
(
    source_tensor,
    target_tensor,
    source_lang_tokenizer,
    target_lang_tokenizer,
) = lth.load_dataset(num_examples)

(
    source_sentences,
    source_validate_sentences,
    target_sentences,
    target_validate_sentences,
) = train_test_split(source_tensor, target_tensor, test_size=0.1)


BUFFER_SIZE = len(source_sentences)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE


# Equivalent to tf.data.Dataset.from_tensor_slices(...).shuffle().batch()
def make_batches(src, tgt, batch_size):
    idx = np.random.permutation(len(src))
    src, tgt = src[idx], tgt[idx]
    n_batch = len(src) // batch_size
    batches = []
    for i in range(n_batch):
        s = i * batch_size
        e = s + batch_size
        batches.append((src[s:e], tgt[s:e]))
    return batches


# ========================================
# Create models
# ========================================

embedding_dim = 256
units = 1024  # GRU dimensionality of the output space.

encoder = Encoder(
    source_lang_tokenizer.vocab_size, embedding_dim, units, BATCH_SIZE
).to(device)
decoder = Decoder(
    target_lang_tokenizer.vocab_size, embedding_dim, units, BATCH_SIZE
).to(device)

# tf.compat.v1.train.AdamOptimizer() equivalent (default lr=0.001)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

# SparseCategoricalCrossentropy(reduction='none') equivalent
loss_object = nn.CrossEntropyLoss(reduction="none")


def loss_function(real, pred):
    # real: (batch, seq_len)
    # pred: (batch, seq_len, vocab_size)
    mask = real != 0  # mask '<pad>' (index=0)

    # CrossEntropyLoss expects (N, C, ...) so permute to (batch, vocab_size, seq_len)
    loss_ = loss_object(pred.permute(0, 2, 1), real)

    mask = mask.to(dtype=loss_.dtype)
    loss_ *= mask
    return loss_.mean()


def compute_accuracy(pred, real):
    # pred: (batch, seq_len, vocab_size)
    # real: (batch, seq_len)
    predicted_ids = pred.argmax(dim=-1)
    mask = real != 0
    correct = (predicted_ids == real) & mask
    return correct.sum().item() / mask.sum().item()


# ========================================
# Training
# ========================================

checkpoint_path = (
    "./checkpoints/seq2seq-attention-"
    + AttentionType
    + "-sample-"
    + str(num_examples)
    + "-embedding-"
    + str(embedding_dim)
    + "-hidden-"
    + str(units)
    + ".pt"
)

if CHECKPOINT and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Latest checkpoint restored!!")


def decoder_training(
    decoder, enc_output, context_vector, target_sentences, target_lang_tokenizer
):

    _dec_hidden = context_vector  # initial attention query = encoder final state

    # Set the initial token.
    # equivalent to: tf.expand_dims([word2idx["<SOS>"]] * BATCH_SIZE, 1)
    sos_idx = target_lang_tokenizer.word2idx["<SOS>"]
    dec_input = torch.full((BATCH_SIZE, 1), sos_idx, dtype=torch.long).to(device)

    predictions = []

    for t in range(1, target_sentences.shape[1]):
        expected_next_token = target_sentences[:, t]  # (batch,)
        prediction, _dec_hidden, _ = decoder(dec_input, _dec_hidden, enc_output)
        # prediction: (batch, vocab_size)

        predictions.append(prediction.unsqueeze(1))  # (batch, 1, vocab_size)

        # Teacher Forcing: feed ground-truth token as next input
        # equivalent to: dec_input = tf.expand_dims(expected_next_token, 1)
        dec_input = expected_next_token.unsqueeze(1).to(dtype=torch.long)  # (batch, 1)

    # Stack along time axis: (batch, seq_len-1, vocab_size)
    predictions = torch.cat(predictions, dim=1)

    return predictions


def train(encoder, decoder, source_sentences, target_sentences, target_lang_tokenizer):

    # numpy -> tensor
    source_sentences = torch.tensor(source_sentences, dtype=torch.long).to(device)
    target_sentences = torch.tensor(target_sentences, dtype=torch.long).to(device)

    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    enc_output, context_vector = encoder(source_sentences)

    predictions = decoder_training(
        decoder, enc_output, context_vector, target_sentences, target_lang_tokenizer
    )

    expected_dec_output = target_sentences[:, 1:]
    loss = loss_function(expected_dec_output, predictions)
    accuracy = compute_accuracy(predictions, expected_dec_output)

    batch_loss = loss / int(target_sentences.shape[1])
    loss.backward()
    optimizer.step()

    return batch_loss.item(), accuracy


#
# To avoid overfitting, limit training duration when using this model.
# In my experience, it performs well after around 7 epochs.
#
# If n_epochs = 0, this model uses the trained parameters saved in the last checkpoint,
# allowing you to perform machine translation without retraining.
if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 7

for epoch in range(1, n_epochs + 1):
    start = time.time()

    total_loss = 0
    total_accuracy = 0.0

    batches = make_batches(source_sentences, target_sentences, BATCH_SIZE)

    for batch, (src_batch, tgt_batch) in enumerate(batches):

        batch_loss, accuracy = train(
            encoder, decoder, src_batch, tgt_batch, target_lang_tokenizer
        )
        total_loss += batch_loss
        total_accuracy += accuracy

        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f} Accuracy: {:.4f}".format(
                    epoch, batch, batch_loss, total_accuracy / (batch + 1)
                )
            )

    if CHECKPOINT:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print("Saving checkpoint for epoch {} at {}".format(epoch, checkpoint_path))

    print("Epoch {}/{} Loss {:.4f}".format(epoch, n_epochs, total_loss / N_BATCH))
    print("Time taken for 1 epoch {:.4f} sec\n".format(time.time() - start))


# ========================================
# Translation
# ========================================


def evaluate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer):

    encoder.eval()
    decoder.eval()

    attention_plot = np.zeros(
        (target_lang_tokenizer.max_length, source_lang_tokenizer.max_length)
    )

    sentence = lth.preprocess_sentence(sentence)
    inputs = source_lang_tokenizer.tokenize(sentence)

    # pad_sequences equivalent
    pad_len = source_lang_tokenizer.max_length - len(inputs)
    inputs = (
        inputs + [0] * pad_len
        if pad_len > 0
        else inputs[: source_lang_tokenizer.max_length]
    )
    inputs = torch.tensor([inputs], dtype=torch.long).to(device)

    result = ""

    with torch.no_grad():
        enc_out, context_vector = encoder(inputs)

        dec_hidden = context_vector  # (1, dec_units): attention query

        # equivalent to: tf.expand_dims([word2idx["<SOS>"]], 0)  shape: (1, 1)
        dec_input = torch.tensor(
            [[target_lang_tokenizer.word2idx["<SOS>"]]], dtype=torch.long
        ).to(device)

        for t in range(target_lang_tokenizer.max_length):
            #
            # Greedy Search
            #
            # predictions  : (1, vocab_size)  <- raw logits from decoder
            # dec_hidden   : (1, dec_units)   <- updated attention query
            # attn_weights : (1, src_seq_len, 1)
            predictions, dec_hidden, attention_weights = decoder(
                dec_input, dec_hidden, enc_out
            )

            # equivalent to: tf.reshape(attention_weights, (-1,)).numpy()
            attention_weights_np = attention_weights.squeeze().cpu().numpy()
            aw_len = min(
                attention_weights_np.shape[0], source_lang_tokenizer.max_length
            )
            attention_plot[t, :aw_len] = attention_weights_np[:aw_len]

            # Apply softmax to logits before argmax (greedy search only)
            # equivalent to: tf.argmax(predictions[0]).numpy()
            probs = torch.softmax(predictions, dim=-1)
            predicted_id = probs[0].argmax().item()

            result += target_lang_tokenizer.idx2word[predicted_id] + " "
            if target_lang_tokenizer.idx2word[predicted_id] == "<EOS>":
                return result, attention_plot

            # equivalent to: tf.expand_dims([predicted_id], 0)  shape: (1, 1)
            dec_input = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return result, attention_plot


def plot_attention_weights(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(attention, cmap=plt.cm.Blues)

    fontdict = {"fontsize": 14}

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(predicted_sentence)))

    ax.set_xticklabels(sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels(predicted_sentence, fontdict=fontdict)

    fig.colorbar(im, ax=ax)
    plt.show()


def translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer):

    result, attention_plot = evaluate(
        sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer
    )

    result = result.capitalize()
    sentence = lth.preprocess_sentence(sentence)
    attention_plot = attention_plot[
        : len(result.split(" ")), : len(sentence.split(" "))
    ]
    plot_attention_weights(attention_plot, sentence.split(" "), result.split(" "))

    return result


"""
# for debug:
# sentence = "No nos gusta la lluvia."
# sentence = "Nos gusta la lluvia."
sentence = "Su voz suena muy bello."
# sentence = "Esta bien."
result = translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer)
print("Input    : {}".format(sentence))
print("Predicted: {}".format(result))

sys.exit()
"""

#
#
#
keys = np.arange(len(source_validate_sentences))
keys = np.random.permutation(keys)[:10]

for i in range(len(keys)):
    print("===== [{}] ======".format(i + 1))
    sentence = source_lang_tokenizer.detokenize(
        source_validate_sentences[i], with_pad=False
    )
    result = translate(
        sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer
    )
    print("Input    : {}".format(sentence))
    print("Predicted: {}".format(result))
    print(
        "Correct  : {}".format(
            target_lang_tokenizer.detokenize(
                target_validate_sentences[i], with_pad=False
            )
        )
    )

print(encoder)
print(decoder)
