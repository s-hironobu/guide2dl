#
# A translator from Spanish to English using Transformer.
#
# This is modified based on https://www.tensorflow.org/text/tutorials/transformer.
#
# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#  tensorflow-macos         2.10.0
#  tensorflow-metal         0.6.
#  scikit-learn             1.2.0
#
#   Copyright (c) 2024-2025, Hironobu Suzuki @ interdb.jp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
import sys

sys.path.append("..")
from Common import LanguageTranslationHelper as lth
from sklearn.model_selection import train_test_split


logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings

CHECKPOINT = True


# ========================================
# Define functions
# ========================================

import platform
import subprocess

def _is_m1_mac():
    # Check if the OS is macOS
    if platform.system() != "Darwin":
        return False
    # Check if the machine is ARM-based (which indicates Apple Silicon)
    machine = platform.machine()
    if machine not in ["arm64", "aarch64"]:
        return False
    # Further verify by checking the hardware model
    try:
        model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if "Apple" in model:
            return True
    except subprocess.CalledProcessError:
        return False
    return False


#
# Positional Encoding
#
def positional_encoding(position, d_model):
    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


#
# Scaled dot product attention
#
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


#
# point-wise feed forward network
#
def point_wise_feed_forward_network(d_model, d_ffn):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_ffn, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ]
    )


# ========================================
# Define Classes
# ========================================

#
# Multi-head attention
#
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, mask):
        def _split_heads(x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        batch_size = tf.shape(q)[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = _split_heads(q, batch_size)
        k = _split_heads(k, batch_size)
        v = _split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


#
# Encoder and Decoder
#


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ffn)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, d_ffn)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, source_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(source_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ffn, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attention_weights = {}

        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, attention_weight = self.enc_layers[i](x, training, mask)
            attention_weights["encoder_layer{}".format(i + 1)] = attention_weight

        return x, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ffn, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        return x, attention_weights


#
# Transformer
#


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ffn, source_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ffn, source_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ffn, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, source_sentences, target_sentences, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output, encoder_attention_weights = self.encoder(source_sentences, training, enc_padding_mask)
        dec_output, decoder_attention_weights = self.decoder(target_sentences, enc_output, training, look_ahead_mask, dec_padding_mask)

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


dataset = tf.data.Dataset.from_tensor_slices(
    (source_tensor_train, target_tensor_train)
).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# ========================================
# Create model
# ========================================

#
# Hyper parameters
#
# Note: This implementation does not include a hyperparameter for explicit sequence length limitation,
#       because this is a toy program and assumes the longest sentence in the received dataset
#       defines the maximum sequence length.

num_layers = 6
d_model = 128
d_ffn = 512
num_heads = 8

dropout_rate = 0.1


#
# Optimizer
#

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

if _is_m1_mac() == True:
    """
    WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs,
    please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
    """
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


#
# Loss function
#

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")


transformer = Transformer(
    num_layers,
    d_model,
    num_heads,
    d_ffn,
    source_lang_tokenizer.vocab_size,
    target_lang_tokenizer.vocab_size,
    pe_input=source_lang_tokenizer.vocab_size,
    pe_target=target_lang_tokenizer.vocab_size,
    rate=dropout_rate,
)


def create_masks(source_sentences, target_sentences):
    def _create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # Expand dimensions to add padding logits
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def _create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    # Padding masks
    enc_padding_mask = _create_padding_mask(source_sentences)
    dec_padding_mask = _create_padding_mask(source_sentences)
    dec_target_padding_mask = _create_padding_mask(target_sentences)

    # Look Ahead mask for masked self-attention layers
    look_ahead_mask = _create_look_ahead_mask(tf.shape(target_sentences)[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# ========================================
# Training
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
)

if CHECKPOINT == True:
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train(source_sentences, target_sentences):

    expected_target_sentences = target_sentences[:, 1:]
    target_sentences = target_sentences[:, :-1]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        source_sentences, target_sentences
    )

    with tf.GradientTape() as tape:
        predictions, _, _ = transformer(source_sentences, target_sentences, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(expected_target_sentences, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(expected_target_sentences, predictions)


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

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (source_sentences, target_sentences)) in enumerate(dataset):
        train(source_sentences, target_sentences)
        if batch % 100 == 0:
            print(
                "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch, batch, train_loss.result(), train_accuracy.result()
                )
            )

    if CHECKPOINT == True:
        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} at {}".format(epoch, ckpt_save_path))

    print(
        "Epoch {}/{} Loss {:.4f} Accuracy {:.4f}".format(
            epoch, n_epochs, train_loss.result(), train_accuracy.result()
        )
    )

    print("Time taken for 1 epoch: {:4f} secs\n".format(time.time() - start))


# ========================================
# Translation
# ========================================


def evaluate(sentence, transformer, source_lang_tokenizer, target_lang_tokenizer):

    sentence = lth.preprocess_sentence(sentence)
    sentence = source_lang_tokenizer.tokenize(sentence)

    encoder_input = tf.expand_dims(sentence, 0)
    decoder_input = [target_lang_tokenizer.word2idx["<SOS>"]]

    output = tf.expand_dims(decoder_input, 0)

    for i in range(target_lang_tokenizer.max_length):
        #
        # Greedy Search
        #
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        predictions, encoder_attention_weights, decoder_attention_weights = transformer(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask,
        )

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if target_lang_tokenizer.idx2word[predicted_id.numpy()[0, 0]] == "<EOS>":
            return (tf.squeeze(output, axis=0), encoder_attention_weights, decoder_attention_weights)

        output = tf.concat([output, predicted_id], axis=-1)

    return (tf.squeeze(output, axis=0), encoder_attention_weights, decoder_attention_weights)


def plot_attention_weights(attention, sentence, result, source_lang_tokenizer, target_lang_tokenizer, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = source_lang_tokenizer.tokenize(sentence)
    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # ax.matshow(attention[head][:-1, :], cmap="viridis")
        ax.matshow(attention[head][:-1, :], cmap=plt.cm.Blues)

        fontdict = {"fontsize": 10}

        source_lang_words = (
            ["<SOS>"]
            + source_lang_tokenizer.detokenize(sentence, return_array=True)
            + ["<EOS>"]
        )
        target_lang_words = target_lang_tokenizer.detokenize(result, return_array=True)

        ax.set_xticks(range(len(source_lang_words)))
        ax.set_yticks(range(len(target_lang_words)))

        ax.set_ylim(len(result) - 1.5, -0.5)
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
    result = result.numpy()
    predicted_sentence = target_lang_tokenizer.detokenize(result)

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
                    target_lang_tokenizer.detokenize(result).lstrip()
                ),
                result,
                target_lang_tokenizer,
                target_lang_tokenizer,
                "decoder_layer{}_block1".format(i),
            )

    if decoder_attention_plot is not None:
        for i in decoder_attention_plot:
            plot_attention_weights(
                decoder_attention_weights,
                lth.preprocess_sentence(sentence, no_tags=True),
                result,
                source_lang_tokenizer,
                target_lang_tokenizer,
                "decoder_layer{}_block2".format(i),
            )

    return predicted_sentence


#
#
#


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


transformer.summary()
transformer.get_config()
