#
# A translator from Spanish to English using Transformer.
#
# This is modified based on https://www.tensorflow.org/text/tutorials/transformer.
#
# Developed environment:
# Python                       3.11.x
# torch                        2.x (MPS support: macOS Apple Silicon / CPU fallback)
# numpy                        1.26.x
# matplotlib                   3.9.x
# scikit-learn                 1.5.x
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import logging
import time
import os
import sys

sys.path.append("..")
from Common import LanguageTranslationHelper as lth
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARNING)

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
# Define functions
# ========================================

#
# Positional Encoding
# equivalent to TF positional_encoding()
#
def positional_encoding(position, d_model):
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = _get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # (1, position, d_model)
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


#
# Scaled dot-product attention
# equivalent to TF scaled_dot_product_attention()
#
def scaled_dot_product_attention(q, k, v, mask):
    # q: (batch, heads, seq_q, depth)
    # k: (batch, heads, seq_k, depth)
    # v: (batch, heads, seq_v, depth)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))   # (batch, heads, seq_q, seq_k)

    dk = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)         # (batch, heads, seq_q, depth)

    return output, attention_weights


#
# Point-wise feed-forward network
# equivalent to TF point_wise_feed_forward_network()
#
def point_wise_feed_forward_network(d_model, d_ffn):
    return nn.Sequential(
        nn.Linear(d_model, d_ffn),
        nn.ReLU(),
        nn.Linear(d_ffn, d_model),
    )


# ========================================
# Define Classes
# ========================================

#
# Multi-Head Attention
# equivalent to TF MultiHeadAttention
#
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def _split_heads(self, x, batch_size):
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)
        # equivalent to TF reshape + transpose(perm=[0, 2, 1, 3])
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        # Note: TF call signature is (v, k, q, mask) — same order here
        batch_size = q.shape[0]

        q = self.Wq(q)   # (batch, seq_q, d_model)
        k = self.Wk(k)   # (batch, seq_k, d_model)
        v = self.Wv(v)   # (batch, seq_v, d_model)

        q = self._split_heads(q, batch_size)   # (batch, heads, seq_q, depth)
        k = self._split_heads(k, batch_size)   # (batch, heads, seq_k, depth)
        v = self._split_heads(v, batch_size)   # (batch, heads, seq_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch, heads, seq_q, depth) -> (batch, seq_q, heads, depth)
        # equivalent to TF transpose(perm=[0, 2, 1, 3])
        scaled_attention = scaled_attention.transpose(1, 2)

        # (batch, seq_q, d_model)
        # equivalent to TF reshape(batch_size, -1, d_model)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)   # (batch, seq_q, d_model)

        return output, attention_weights


#
# Encoder Layer
# equivalent to TF EncoderLayer
#
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ffn)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask, training=True):
        # Self-attention (v=x, k=x, q=x)
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output) if training else attn_output
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights


#
# Decoder Layer
# equivalent to TF DecoderLayer
#
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, d_ffn)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, training=True):
        # Block1: Masked self-attention (v=x, k=x, q=x)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1) if training else attn1
        out1 = self.layernorm1(attn1 + x)

        # Block2: Cross-attention (v=enc_output, k=enc_output, q=out1)
        # equivalent to TF: self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2) if training else attn2
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output) if training else ffn_output
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


#
# Encoder
# equivalent to TF Encoder
#
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 source_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(source_vocab_size, d_model)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        # positional encoding: (1, maximum_position_encoding, d_model)
        # registered as buffer so it moves with .to(device)
        self.register_buffer(
            "pos_encoding",
            positional_encoding(maximum_position_encoding, d_model),
        )

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ffn, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, mask, training=True):
        attention_weights = {}

        seq_len = x.shape[1]
        x = self.embedding(x)                                          # (batch, seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]                        # add positional encoding
        x = self.dropout(x) if training else x

        for i in range(self.num_layers):
            x, attention_weight = self.enc_layers[i](x, mask, training)
            attention_weights["encoder_layer{}".format(i + 1)] = attention_weight

        return x, attention_weights


#
# Decoder
# equivalent to TF Decoder
#
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        self.register_buffer(
            "pos_encoding",
            positional_encoding(maximum_position_encoding, d_model),
        )

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ffn, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, training=True):
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)                                          # (batch, seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x) if training else x

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask, training
            )
            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        return x, attention_weights


#
# Transformer
# equivalent to TF Transformer
#
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ffn,
                 source_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ffn,
                               source_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ffn,
                               target_vocab_size, pe_target, rate)

        # Raw logits output — no softmax here.
        # TF uses from_logits=True in SparseCategoricalCrossentropy,
        # so CrossEntropyLoss (which applies log_softmax internally) is the correct equivalent.
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, source_sentences, target_sentences,
                enc_padding_mask, look_ahead_mask, dec_padding_mask, training=True):

        enc_output, encoder_attention_weights = self.encoder(
            source_sentences, enc_padding_mask, training
        )
        dec_output, decoder_attention_weights = self.decoder(
            target_sentences, enc_output, look_ahead_mask, dec_padding_mask, training
        )

        # (batch, seq_len, target_vocab_size)  raw logits
        final_output = self.final_layer(dec_output)

        return final_output, encoder_attention_weights, decoder_attention_weights


# ========================================
# Create dataset
# ========================================

num_examples = 110000
(
    source_tensor,
    target_tensor,
    source_lang_tokenizer,
    target_lang_tokenizer,
) = lth.load_dataset(num_examples)

(
    source_tensor_train,
    source_tensor_val,
    target_tensor_train,
    target_tensor_val,
) = train_test_split(source_tensor, target_tensor, test_size=0.1)

BUFFER_SIZE = len(source_tensor_train)
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

num_layers = 6
d_model = 128
d_ffn = 512
num_heads = 8
dropout_rate = 0.1

#
# Custom learning rate schedule (warmup)
# equivalent to TF CustomSchedule
#
class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def __call__(self):
        self._step += 1
        step = self._step
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(arg1, arg2)


transformer = Transformer(
    num_layers, d_model, num_heads, d_ffn,
    source_lang_tokenizer.vocab_size,
    target_lang_tokenizer.vocab_size,
    pe_input=source_lang_tokenizer.vocab_size,
    pe_target=target_lang_tokenizer.vocab_size,
    rate=dropout_rate,
).to(device)

lr_schedule = CustomSchedule(d_model)

# Adam with warmup: beta_1=0.9, beta_2=0.98, epsilon=1e-9 (same as TF)
optimizer = optim.Adam(
    transformer.parameters(), lr=lr_schedule(), betas=(0.9, 0.98), eps=1e-9
)


def update_lr(optimizer, lr_schedule):
    """Update learning rate according to CustomSchedule each step."""
    lr = lr_schedule()
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


#
# Mask functions
# equivalent to TF create_masks()
#
def create_padding_mask(seq):
    # seq: (batch, seq_len)
    # returns: (batch, 1, 1, seq_len)  float32
    mask = (seq == 0).float()
    return mask[:, None, None, :]   # expand for broadcasting over heads


def create_look_ahead_mask(size):
    # Upper triangular mask (excluding diagonal) — same as TF
    # TF: 1 - band_part(ones, -1, 0)  => upper triangle (excluding diag) = 1
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask   # (size, size)


def create_masks(source_sentences, target_sentences):
    # source_sentences: (batch, src_seq_len)  numpy or tensor
    if not isinstance(source_sentences, torch.Tensor):
        source_sentences = torch.tensor(source_sentences, dtype=torch.long)
    if not isinstance(target_sentences, torch.Tensor):
        target_sentences = torch.tensor(target_sentences, dtype=torch.long)

    enc_padding_mask = create_padding_mask(source_sentences).to(device)
    dec_padding_mask  = create_padding_mask(source_sentences).to(device)

    look_ahead_mask = create_look_ahead_mask(target_sentences.shape[1]).to(device)
    dec_target_padding_mask = create_padding_mask(target_sentences).to(device)

    # combined_mask = max(dec_target_padding_mask, look_ahead_mask)
    # equivalent to TF: tf.maximum(dec_target_padding_mask, look_ahead_mask)
    combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


#
# Loss function
# equivalent to TF loss_function() with SparseCategoricalCrossentropy(from_logits=True)
#
loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=0)


def loss_function(real, pred):
    # real: (batch, seq_len)
    # pred: (batch, seq_len, vocab_size)  raw logits
    # CrossEntropyLoss expects (N, C, d) so permute
    loss_ = loss_object(pred.permute(0, 2, 1), real)
    return loss_.mean()


def compute_accuracy(pred, real):
    # pred: (batch, seq_len, vocab_size)  raw logits
    # real: (batch, seq_len)
    predicted_ids = pred.argmax(dim=-1)
    mask = real != 0
    correct = (predicted_ids == real) & mask
    return correct.sum().item() / mask.sum().item()


# ========================================
# Checkpoint
# ========================================

checkpoint_path = (
    "./checkpoints/transformer-sample-"
    + str(num_examples)
    + "-layers-"
    + str(num_layers)
    + "-d_model-"
    + str(d_model)
    + "-d_ffn-"
    + str(d_ffn)
    + "-heads-"
    + str(num_heads)
    + ".pt"
)

if CHECKPOINT and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    transformer.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_schedule._step = ckpt.get("lr_step", 0)
    print("Latest checkpoint restored!!")


# ========================================
# Training
# ========================================

def train(source_sentences, target_sentences):
    # numpy -> tensor
    source_sentences = torch.tensor(source_sentences, dtype=torch.long).to(device)
    target_sentences = torch.tensor(target_sentences, dtype=torch.long).to(device)

    # Teacher Forcing:
    # input to decoder  : target[:, :-1]  (drop last token)
    # expected output   : target[:, 1:]   (drop first <SOS> token)
    expected_target_sentences = target_sentences[:, 1:]
    target_sentences_in       = target_sentences[:, :-1]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        source_sentences, target_sentences_in
    )

    transformer.train()
    optimizer.zero_grad()

    predictions, _, _ = transformer(
        source_sentences, target_sentences_in,
        enc_padding_mask, combined_mask, dec_padding_mask,
        training=True,
    )

    loss = loss_function(expected_target_sentences, predictions)
    acc  = compute_accuracy(predictions, expected_target_sentences)

    loss.backward()

    # Update LR according to warmup schedule (per step, same as TF)
    update_lr(optimizer, lr_schedule)
    optimizer.step()

    return loss.item(), acc


#
# Set n_epochs at least 10 when you do training.
#
# If n_epochs = 0, this model uses the trained parameters saved in the last checkpoint,
# allowing you to perform machine translation without retraining.
if len(sys.argv) == 2:
    n_epochs = int(sys.argv[1])
else:
    n_epochs = 10

for epoch in range(1, n_epochs + 1):
    start = time.time()

    total_loss = 0.0
    total_acc  = 0.0
    batches = make_batches(source_tensor_train, target_tensor_train, BATCH_SIZE)

    for batch, (src_batch, tgt_batch) in enumerate(batches):
        loss, acc = train(src_batch, tgt_batch)
        total_loss += loss
        total_acc  += acc

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                epoch, batch, total_loss / (batch + 1), total_acc / (batch + 1)
            ))

    if CHECKPOINT:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict":     transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_step":              lr_schedule._step,
            },
            checkpoint_path,
        )
        print("Saving checkpoint for epoch {} at {}".format(epoch, checkpoint_path))

    print("Epoch {}/{} Loss {:.4f} Accuracy {:.4f}".format(
        epoch, n_epochs, total_loss / N_BATCH, total_acc / N_BATCH
    ))
    print("Time taken for 1 epoch: {:4f} secs\n".format(time.time() - start))


# ========================================
# Translation
# ========================================

def evaluate(sentence, transformer, source_lang_tokenizer, target_lang_tokenizer):

    transformer.eval()

    sentence = lth.preprocess_sentence(sentence)
    sentence = source_lang_tokenizer.tokenize(sentence)

    # encoder_input: (1, src_seq_len)
    # equivalent to TF: tf.expand_dims(sentence, 0)
    encoder_input = torch.tensor([sentence], dtype=torch.long).to(device)

    # Start decoder with <SOS> token
    # equivalent to TF: output = tf.expand_dims([word2idx["<SOS>"]], 0)
    decoder_input = [target_lang_tokenizer.word2idx["<SOS>"]]
    output = torch.tensor([decoder_input], dtype=torch.long).to(device)   # (1, 1)

    with torch.no_grad():
        for i in range(target_lang_tokenizer.max_length):
            #
            # Greedy Search
            #
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output
            )

            predictions, encoder_attention_weights, decoder_attention_weights = transformer(
                encoder_input, output,
                enc_padding_mask, combined_mask, dec_padding_mask,
                training=False,
            )

            # Take the last token's prediction and apply softmax to get probabilities
            # equivalent to TF: predictions = predictions[:, -1:, :]
            predictions_last = predictions[:, -1:, :]                  # (1, 1, vocab_size)
            probs = torch.softmax(predictions_last, dim=-1)
            predicted_id = probs.argmax(dim=-1).to(torch.long)          # (1, 1)

            # equivalent to TF: idx2word[predicted_id.numpy()[0, 0]] == "<EOS>"
            if target_lang_tokenizer.idx2word[predicted_id[0, 0].item()] == "<EOS>":
                return (
                    output.squeeze(0),
                    encoder_attention_weights,
                    decoder_attention_weights,
                )

            # Append predicted token to output sequence
            # equivalent to TF: output = tf.concat([output, predicted_id], axis=-1)
            output = torch.cat([output, predicted_id], dim=-1)          # (1, seq+1)

    return (output.squeeze(0), encoder_attention_weights, decoder_attention_weights)


def plot_attention_weights(
    attention, sentence, result, source_lang_tokenizer, target_lang_tokenizer, layer
):
    fig = plt.figure(figsize=(16, 8))

    sentence = source_lang_tokenizer.tokenize(sentence)

    # attention[layer]: (batch, heads, seq_q, seq_k) -> squeeze batch -> (heads, seq_q, seq_k)
    # equivalent to TF: tf.squeeze(attention[layer], axis=0)
    attn = attention[layer].squeeze(0).cpu().numpy()

    for head in range(attn.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)
        ax.matshow(attn[head][:-1, :], cmap=plt.cm.Blues)

        fontdict = {"fontsize": 10}

        source_lang_words = (
            ["<SOS>"]
            + source_lang_tokenizer.detokenize(sentence, return_array=True)
            + ["<EOS>"]
        )
        result_np = result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        target_lang_words = target_lang_tokenizer.detokenize(result_np, return_array=True)

        ax.set_xticks(range(len(source_lang_words)))
        ax.set_yticks(range(len(target_lang_words)))

        ax.set_ylim(len(result_np) - 1.5, -0.5)
        ax.set_xticklabels(source_lang_words, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(target_lang_words, fontdict=fontdict)
        ax.set_xlabel("Head {}".format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(
    sentence,
    transformer,
    source_lang_tokenizer,
    target_lang_tokenizer,
    encoder_self_attention_plot=None,
    decoder_self_attention_plot=None,
    decoder_attention_plot=None,
):
    result, encoder_attention_weights, decoder_attention_weights = evaluate(
        sentence, transformer, source_lang_tokenizer, target_lang_tokenizer
    )

    result_np = result.cpu().numpy()
    predicted_sentence = target_lang_tokenizer.detokenize(result_np)

    if encoder_self_attention_plot is not None:
        for i in encoder_self_attention_plot:
            plot_attention_weights(
                encoder_attention_weights,
                lth.preprocess_sentence(sentence, no_tags=True),
                source_lang_tokenizer.tokenize(
                    lth.preprocess_sentence(sentence.lstrip(), no_tags=True)
                ),
                source_lang_tokenizer,
                source_lang_tokenizer,
                "encoder_layer{}".format(i),
            )

    if decoder_self_attention_plot is not None:
        for i in decoder_self_attention_plot:
            plot_attention_weights(
                decoder_attention_weights,
                lth.preprocess_sentence(
                    target_lang_tokenizer.detokenize(result_np).lstrip()
                ),
                result_np,
                target_lang_tokenizer,
                target_lang_tokenizer,
                "decoder_layer{}_block1".format(i),
            )

    if decoder_attention_plot is not None:
        for i in decoder_attention_plot:
            plot_attention_weights(
                decoder_attention_weights,
                lth.preprocess_sentence(sentence, no_tags=True),
                result_np,
                source_lang_tokenizer,
                target_lang_tokenizer,
                "decoder_layer{}_block2".format(i),
            )

    return predicted_sentence


"""
# for debug:
sentence = "me gustaria oir que tienes que decir sobre esto.."
result = translate(sentence, transformer, source_lang_tokenizer, target_lang_tokenizer,
                   encoder_self_attention_plot=[1, 2],
                   decoder_self_attention_plot=[1, 2],
                   decoder_attention_plot=[1, 2])
print("Input    : {}".format(sentence))
print("Predicted: {}".format(result))

sys.exit()
"""

keys = np.arange(len(source_tensor_val))
keys = np.random.permutation(keys)[:10]

for i in range(len(keys)):
    print("===== [{}] ======".format(i + 1))
    sentence = source_lang_tokenizer.detokenize(source_tensor_val[i], with_pad=False)

    result = translate(
        sentence,
        transformer,
        source_lang_tokenizer,
        target_lang_tokenizer,
        encoder_self_attention_plot=[1, 2],
        decoder_self_attention_plot=[1],
        decoder_attention_plot=[1],
    )
    print("Input    : {}".format(sentence))
    print("Predicted: {}".format(result))
    print("Correct  : {}".format(target_lang_tokenizer.detokenize(target_tensor_val[i], with_pad=False)))

print(transformer)
