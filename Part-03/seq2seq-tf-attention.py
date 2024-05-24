#
# A translator from Spanish to English using seq2seq with Attention.
#
# This is modified based on https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ja
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
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp


from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Attention
import logging
import time
import sys
sys.path.append("..")
from Common import LanguageTranslationHelper as lth

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings

#
# Use/Write CHECKPOINT data or not.
#

CHECKPOINT = True

#
# Select Attention Type.
#

AttentionType = "Bahdanau"  # a.k.a. Additive attention, Multi-Layer perceptron.
# AttentionType = "Luong"  # a.k.a. Bilinear , General attention.

# ========================================
# Define Classes
# ========================================

#
# Encoder
#
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        """
        Bahdanau attention model uses bidirectional RNN in the original paper,
        but we use a simple GRU.
        """
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state


#
# Decoder
#
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.softmax = tf.keras.layers.Dense(vocab_size, activation="softmax")

        if AttentionType == "Bahdanau":
            self.attention = Attention.BahdanauAttention(self.dec_units)
        elif AttentionType == "Luong":
            self.attention = Attention.LuongAttention(self.dec_units)
            self.W = tf.keras.layers.Dense(2 * units)
        else:
            print("Error: {} is not supported.".format(Attention))
            sys.exit()

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)

        if AttentionType == "Bahdanau":
            context_vector, attention_weights = self.attention(hidden, enc_output)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(inputs=x)
            output = tf.reshape(output, (-1, output.shape[2]))
        else:
            """ Luong """
            output, state = self.gru(inputs=x, initial_state=hidden)
            context_vector, attention_weights = self.attention(state, enc_output)
            output = tf.nn.tanh(self.W(tf.concat([state, context_vector], axis=-1)))

        x = self.softmax(output)

        return x, state, attention_weights


# ========================================
# Create Dataset
# ========================================

num_examples = 110000
(
    source_tensor,
    target_tensor,
    source_lang_tokenizer,
    target_lang_tokenizer
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

dataset = tf.data.Dataset.from_tensor_slices(
    (source_sentences, target_sentences)
).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# ========================================
# Create models
# ========================================

embedding_dim = 256
units = 1024  # GRU dimensionality of the output space.

encoder = Encoder(source_lang_tokenizer.vocab_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(target_lang_tokenizer.vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.compat.v1.train.AdamOptimizer()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) # this masks '<pad>'
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
    "./checkpoints/seq2seq-attention-"
    + AttentionType
    + "-sample-"
    + str(num_examples)
    + "-embedding-"
    + str(embedding_dim)
    + "-hidden-"
    + str(units)
)

if CHECKPOINT == True:
    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")


@tf.function
def decoder_training(decoder, enc_output, context_vector, target_sentences, target_lang_tokenizer):

    dec_hidden = context_vector

    # Set the initial token.
    dec_input = tf.expand_dims([target_lang_tokenizer.word2idx["<SOS>"]] * BATCH_SIZE, 1)

    #
    for t in range(1, target_sentences.shape[1]):
        expected_next_token = target_sentences[:, t]
        prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        if t == 1:
            predictions = tf.concat([tf.expand_dims(prediction, axis=1)], axis=1)
        else:
            predictions = tf.concat([predictions, tf.expand_dims(prediction, axis=1)], axis=1)

        # Retrieve the next token.
        dec_input = tf.expand_dims(expected_next_token, 1)

    return predictions


@tf.function
def train(encoder, decoder, source_sentences, target_sentences, target_lang_tokenizer):

    with tf.GradientTape() as tape:
        enc_output, context_vector = encoder(source_sentences)

        predictions = decoder_training(decoder, enc_output, context_vector, target_sentences, target_lang_tokenizer)

        expected_dec_output = target_sentences[:, 1:]
        loss = loss_function(expected_dec_output, predictions)
        train_accuracy(expected_dec_output, predictions)

    batch_loss = loss / int(target_sentences.shape[1])
    variables = encoder.variables + decoder.variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

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
    train_accuracy.reset_states()

    for (batch, (source_sentences, target_sentences)) in enumerate(dataset):

        batch_loss = train(encoder, decoder, source_sentences, target_sentences, target_lang_tokenizer)
        total_loss += batch_loss

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f} Accuracy: {:.4f}".format(epoch, batch, batch_loss.numpy(), train_accuracy.result()))

    if CHECKPOINT == True:
        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} at {}".format(epoch, ckpt_save_path))


    print("Epoch {}/{} Loss {:.4f}".format(epoch, n_epochs, total_loss / N_BATCH))
    print("Time taken for 1 epoch {:.4f} sec\n".format(time.time() - start))



# ========================================
# Translation
# ========================================

def evaluate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer):

    attention_plot = np.zeros((target_lang_tokenizer.max_length, source_lang_tokenizer.max_length))
    sentence = lth.preprocess_sentence(sentence)
    inputs = source_lang_tokenizer.tokenize(sentence)
    inputs = tf.compat.v1.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=source_lang_tokenizer.max_length, padding="post")
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    enc_out, context_vector = encoder(inputs)

    dec_hidden = context_vector
    dec_input = tf.expand_dims([target_lang_tokenizer.word2idx["<SOS>"]], 0)

    for t in range(target_lang_tokenizer.max_length):
        #
        # Greedy Search
        #
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_lang_tokenizer.idx2word[predicted_id] + " "
        if target_lang_tokenizer.idx2word[predicted_id] == "<EOS>":
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, attention_plot


def plot_attention_weights(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # im = ax.matshow(attention, cmap="viridis")
    im = ax.matshow(attention, cmap=plt.cm.Blues)

    fontdict = {"fontsize": 14}

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(predicted_sentence)))

    ax.set_xticklabels(sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels(predicted_sentence, fontdict=fontdict)

    fig.colorbar(im, ax=ax)
    plt.show()


def translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer):

    result, attention_plot = evaluate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer)

    result = result.capitalize()
    sentence = lth.preprocess_sentence(sentence)
    attention_plot = attention_plot[: len(result.split(" ")), : len(sentence.split(" "))]
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
    sentence = source_lang_tokenizer.detokenize(source_validate_sentences[i], with_pad=False)
    result = translate(sentence, encoder, decoder, source_lang_tokenizer, target_lang_tokenizer)
    print("Input    : {}".format(sentence))
    print("Predicted: {}".format(result))
    print("Correct  : {}".format(target_lang_tokenizer.detokenize(target_validate_sentences[i], with_pad=False)))

encoder.summary()
decoder.summary()
decoder.get_config()
