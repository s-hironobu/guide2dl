#
# A translator from Spanish to English using seq2seq.
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

        # return_sequences=False, return_state=False equivalent:
        #   nn.GRU returns (output, h_n)
        #     output: (batch, seq_len, enc_units)  <- all time steps
        #     h_n   : (1, batch, enc_units)        <- final hidden state
        #   TF GRU with return_sequences=False returns only the final hidden state.
        #   That corresponds to h_n.squeeze(0): (batch, enc_units)
        _, h_n = self.gru(x)
        output = h_n.squeeze(0)  # (batch, enc_units)
        return output


#
# Decoder
#
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=dec_units,
            batch_first=True,
        )
        # recurrent_initializer="glorot_uniform" equivalent
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        self.softmax = nn.Linear(dec_units, vocab_size)

    def forward(self, x, hidden):
        # x     : (batch, seq_len)
        # hidden: (batch, dec_units)  <- encoder output or previous decoder state

        x = self.embedding(x)

        # initial_state equivalent:
        #   TF GRU accepts hidden as initial_state directly (batch, units).
        #   PyTorch GRU expects h_0 shape (num_layers, batch, hidden_size),
        #   so we unsqueeze(0) to add the num_layers dimension.
        # .contiguous() ensures correct memory layout on MPS/CUDA devices.
        hidden_h0 = hidden.unsqueeze(0).contiguous()  # (1, batch, dec_units)

        # return_sequences=True, return_state=True equivalent:
        #   output: (batch, seq_len, dec_units)  <- all time steps
        #   h_n   : (1, batch, dec_units)        <- final hidden state
        output, h_n = self.gru(x, hidden_h0)

        # state: (batch, dec_units)  <- squeeze out num_layers dimension
        # .contiguous() ensures the state tensor is safe to pass back as hidden
        state = h_n.squeeze(0).contiguous()

        # Output raw logits — do NOT apply softmax here.
        # nn.CrossEntropyLoss applies log_softmax internally.
        # Softmax is applied only in evaluate() for greedy search.
        output = self.softmax(output)  # (batch, seq_len, vocab_size)  <- raw logits

        return output, state


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
# Create model
# ========================================

embedding_dim = 256
units = 1024  # LSTM/GRU dimensionality of the output space.

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
    #
    # mask '<pad>' (index=0), same as TF version
    mask = real != 0

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
    "./checkpoints/seq2seq-sample-"
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


def train(encoder, decoder, source_sentences, target_sentences, target_lang_tokenizer):

    # numpy -> tensor
    source_sentences = torch.tensor(source_sentences, dtype=torch.long).to(device)
    target_sentences = torch.tensor(target_sentences, dtype=torch.long).to(device)

    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    context_vector = encoder(source_sentences)

    # Input sentences:           e.g. ['<sos>', 'this', 'is', 'a', 'pen', '.', '<eos>']
    dec_input = target_sentences[:, :-1]
    # Expected output sentences: e.g. ['this', 'is', 'a', 'pen', '.', '<eos>', '<pad>']
    expected_dec_output = target_sentences[:, 1:]

    predictions, _ = decoder(dec_input, context_vector)
    loss = loss_function(expected_dec_output, predictions)
    accuracy = compute_accuracy(predictions, expected_dec_output)

    batch_loss = loss / int(target_sentences.shape[1])
    loss.backward()
    optimizer.step()

    return batch_loss.item(), accuracy


#
# Set n_epochs at least 20 when you do training.
#
# If n_epochs = 0, this model uses the trained parameters saved in the last checkpoint,
# allowing you to perform machine translation without retraining.
if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 10


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
        context_vector = encoder(inputs)

        dec_hidden = context_vector
        # tf.expand_dims([target_lang_tokenizer.word2idx["<SOS>"]], 0) equivalent
        dec_input = torch.tensor(
            [[target_lang_tokenizer.word2idx["<SOS>"]]], dtype=torch.long
        ).to(device)

        for t in range(target_lang_tokenizer.max_length):
            #
            # Greedy Search
            #
            # dec_input  : (1, 1)              <- batch=1, seq_len=1
            # predictions: (1, 1, vocab_size)  <- raw logits
            # dec_hidden : (1, dec_units)      <- updated hidden state for next step
            predictions, dec_hidden = decoder(dec_input, dec_hidden)

            # Apply softmax to logits before argmax (greedy search only)
            # equivalent to tf.argmax(predictions[0][0]).numpy()
            probs = torch.softmax(predictions, dim=-1)
            predicted_id = probs[0, 0, :].argmax().item()

            result += target_lang_tokenizer.idx2word[predicted_id] + " "
            if target_lang_tokenizer.idx2word[predicted_id] == "<EOS>":
                return result

            # Feed predicted token as next input: (1, 1)
            # equivalent to tf.expand_dims([predicted_id], 0)
            dec_input = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return result


def translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer):
    result = evaluate(
        sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer
    )
    return result.capitalize()


"""
# for debug:
#sentence = "Su voz suena muy bello."
#sentence = "No nos gusta la lluvia."
sentence = "Nos gusta la lluvia."
result = translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer)
print("Input    : {}".format(sentence))
print("Predicted: {}".format(result))

sys.exit()
"""
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
