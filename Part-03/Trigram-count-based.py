#
# Tri-gram.
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

import matplotlib.pyplot as plt
import numpy as np
import Tokenizer as tokenizer
import sys

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

Lang =tokenizer.Tokenizer(file_path)

# Add two <sos> and <eos> before and after the sentences for consistent trigram calculations.
# E.g. "<SOS> <SOS> Go for it . <EOS> <EOS>" => P("Go"| <SOS>,<SOS>), P("for"| "Go",<SOS>), etc.
Lang.read_dataset(add_sos=2)
vocab_size = Lang.vocab_size()


print("=================================")
print("Using Trigram")
print("vocabulary size: {}".format(vocab_size))
print("number of sentences: {}".format(Lang.num_texts()))
print("=================================")

# ========================================
# Compute transition matrix
# ========================================

transition_matrix = np.zeros((vocab_size, vocab_size, vocab_size))

for _, sentence in enumerate(Lang.sentences):
    for i in range(len(sentence) - 2):
        idx1 = sentence[i]
        idx2 = sentence[i + 1]
        idx3 = sentence[i + 2]
        transition_matrix[idx1][idx2][idx3] += 1

for i in range(vocab_size):
    for j in range(vocab_size):
        _sum = np.sum(transition_matrix[i][j])
        if _sum > 0:
            transition_matrix[i][j] /= _sum

# ========================================
# Test
# ========================================

# Because of Tri-gram
n_sequence = 2


def check_length(keys, n_sequence):
    for key in keys:
        if len(Lang.sentences[key]) <= n_sequence:
            return False
    return True


num_candidate = 5

while True:
    keys = np.arange(Lang.num_texts())
    n = np.minimum(num_candidate, len(keys))
    keys = np.random.permutation(keys)[:n]
    if check_length(keys, n_sequence):
        break

words = []
vals = []

for key in keys:
    print("Text:{}".format(Lang.texts[i]))

    # Get the indexes of the second and third words from the end.
    # e.g. sentences[key][-1] := <eos>, sentences[key][-2] := <eos>, sentences[i][-3] := last word.
    idx1 = Lang.sentences[key][-5]
    idx2 = Lang.sentences[key][-4]

    cp = transition_matrix[idx1][idx2].copy()
    cp[Lang.word2idx['<eos>']] = 0.0
    sorted_list = sorted(cp, reverse=True)

    w = []
    v = []
    for _ in range(num_candidate):
        _max_idx = np.argmax(cp)
        _max_val = np.max(cp)
        w.append(Lang.idx2word[_max_idx])
        v.append(_max_val)

        cp[_max_idx] = 0.0

    words.append(w)
    vals.append(v)

    print("Predicted last word:")
    for i in range(n):
        if v[i] > 0.0:
            print("\t{}\t=> {:>5f}".format(w[i], v[i]))
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
