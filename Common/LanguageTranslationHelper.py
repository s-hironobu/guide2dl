#
# Helper modules for language translation.
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

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import unicodedata
import re
import os
import numpy as np

class Tokenizer:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def set_max_length(self, max_length):
        self.max_length = max_length

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(" "))

        self.vocab = sorted(self.vocab)
        # replace <SOS> and <EOS> to the tail of vocab.
        self.vocab.remove('<SOS>')
        self.vocab.remove('<EOS>')
        self.vocab.append('<SOS>')
        self.vocab.append('<EOS>')

        """
        The index of "<pad>" must be 0, because the loss_function in seq2seq-tf.py,
        seq2seq-tf-attention.py and Transformer-tf.py assume it.
        """
        self.word2idx["<pad>"] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

        # len(word2idx) = len(vacab) + 1, since word2idx contains '<pad>'.
        self.vocab_size = len(self.word2idx)

    def padding(self, wordlist):
        for i in range(len(wordlist)):
            for j in range(self.max_length - len(wordlist[i])):
                wordlist[i].append('0')
        return np.array(wordlist, dtype=int)

    def tokenize(self, sentence):
        return [self.word2idx[key] for key in sentence.split(" ")]

    def detokenize(self, tensors, return_array=False, with_pad=True, with_sos=False):
        if return_array == False:
            sentence = ""
        else:
            sentence = []
        for i in range(len(tensors)):
            if tensors[i] < self.vocab_size:
                w = self.idx2word[tensors[i]]
                if i == 0:
                    if with_sos == False and w == '<SOS>':
                        # remove '<SOS>'
                        continue
                    if w != '<SOS>':
                        w = w.capitalize()
                    if return_array == False:
                        sentence = w
                    else:
                        sentence.append(w)
                elif i == 1 and with_sos == True:
                    if return_array == False:
                        sentence += ' ' + w.capitalize()
                    else:
                        sentence.append(w.capitalize())

                elif w == ',' or w == '.':
                    if return_array == False:
                        sentence += w
                    else:
                        sentence.append(w)

                else:
                    if with_sos == False and w == '<EOS>':
                        # remove '<EOS>'
                        continue
                    elif with_pad == False and w == '<pad>':
                        # remove '<pad>'
                        continue
                    else:
                        if return_array == False:
                            sentence += ' ' + w
                        else:
                            sentence.append(w)

        return sentence


def preprocess_sentence(w, no_tags=False):
    def __unicode_to_ascii(s):
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    w = __unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z0-9'?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    if no_tags == False:
        w = "<SOS> " + w + " <EOS>"
    return w


#def load_dataset(num_examples, no_tags=False, same_length=False):
def load_dataset(num_examples, no_tags=False):
    def __max_length(tensor):
        return max(len(t) for t in tensor)

    def __create_dataset(path, num_examples):
        lines = open(path, encoding="UTF-8").read().strip().split("\n")
        word_pairs = [
            [preprocess_sentence(w, no_tags) for w in l.split("\t")]
            for l in lines[:num_examples]
        ]
        return word_pairs

    try:
        path_to_zip = tf.keras.utils.get_file(
            "spa-eng.zip",
            origin="http://download.tensorflow.org/data/spa-eng.zip",
            extract=True,
        )
    except:
        path_to_zip = '../DataSets/keras/'

    # Read from ~/.keras/datasets/spa-eng/ if possible, otherwise, DataSets/keras.
    path = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
    pairs = __create_dataset(path, num_examples)

    source_lang_tokenizer = Tokenizer(sp for en, sp in pairs)
    target_lang_tokenizer = Tokenizer(en for en, sp in pairs)

    source_sentence = [[source_lang_tokenizer.word2idx[s] for s in sp.split(" ")] for en, sp in pairs]
    target_sentence = [[target_lang_tokenizer.word2idx[s] for s in en.split(" ")] for en, sp in pairs]

    source_lang_tokenizer.set_max_length(__max_length(source_sentence))
    target_lang_tokenizer.set_max_length(__max_length(target_sentence))

    """
    if same_length == True:
        max_len = max(max_length_source, max_length_target)
        max_length_source, max_length_target = max_len, max_len
    """

    source_sentence = source_lang_tokenizer.padding(source_sentence)
    target_sentence = target_lang_tokenizer.padding(target_sentence)

    return source_sentence, target_sentence, source_lang_tokenizer, target_lang_tokenizer


if __name__ == '__main__':

    num_examples = 22
    (
        source_sentence,
        target_sentence,
        source_lang_tokenizer,
        target_lang_tokenizer
    ) = load_dataset(num_examples)


    print("==== Vocabulary ====")
    print("-----------------------------")
    print("Source language Vocabulary")
    print("-----------------------------")
    for i in range(len(source_lang_tokenizer.idx2word)):
        print("{:>2} {}".format(i, source_lang_tokenizer.idx2word[i]))

    word = 'hola'
    idx = source_lang_tokenizer.word2idx[word]
    print("\nsource_lang_tokenizer.word2idx['{}'] = {}".format(word, idx))
    print("source_lang_tokenizer.idx2word[{}] = {}".format(idx, source_lang_tokenizer.idx2word[idx]))

    print("-----------------------------")
    print("Target language Vocabulary")
    print("-----------------------------")
    for i in range(len(target_lang_tokenizer.idx2word)):
        print("{:>2} {}".format(i, target_lang_tokenizer.idx2word[i]))

    word = 'hi'
    idx = target_lang_tokenizer.word2idx[word]
    print("\ntarget_lang_tokenizer.word2idx['{}'] = {}".format(word, idx))
    print("target_lang_tokenizer.idx2word[{}] = {}".format(idx, target_lang_tokenizer.idx2word[idx]))

    print("==== Max Length ====")
    print("max_length_source = {}, max_length_target = {}".format(source_lang_tokenizer.max_length, target_lang_tokenizer.max_length))

    print("==== Sentences ====")
    print("source_sentence.shape = ", source_sentence.shape)
    print("target_sentence.shape = ", target_sentence.shape)

    print("---- Source Sentences ----")
    for i in range(num_examples - 9, num_examples):
        print("source_sentence[{}] = {} = {}".format(i, source_sentence[i], source_lang_tokenizer.detokenize(source_sentence[i], with_sos=True)))

    print("---- Target Sentences ----")
    for i in range(num_examples - 9, num_examples):
        print("target_sentence[{}] = {} = {}".format(i, target_sentence[i], target_lang_tokenizer.detokenize(target_sentence[i], with_sos=True)))


    """
    embedding_layer = layers.Embedding(100, 5, embeddings_initializer=keras.initializers.RandomNormal(seed=1))

    result = embedding_layer(tf.constant([1,0,3]))
    result.numpy()

    print("result=", result)
    """
