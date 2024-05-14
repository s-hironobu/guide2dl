#
# Subset of the LanguageTranslationHelper.py@Common, which is a module for language translation.
#
# Developed environment:
#  Python                   3.9.13
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

class Tokenizer:
    def __init__(self, file_name, category=False):
        self.word2idx = {}
        self.idx2word = {}
        self.sentences = []
        self.labels = []
        self.__vocab = set()
        self.__file_name = file_name
        self.__category = category

    def read_dataset(self, padding=False, add_sos=1):
        self.__padding = padding
        self.__add_sos = add_sos
        self._max_len = 0

        # Read sentences
        file_text = open(self.__file_name, "r")
        self.texts = file_text.readlines()
        file_text.close()

        if self.__category == False:
            self.__textA = self.texts.copy()
        else:
            _texts = self.texts.copy()
            self.__textA = []

            for _text in _texts:
                _tmp = _text.rsplit("\t", maxsplit=1)
                self.__textA.append(_tmp[0])
                self.labels.append(int(_tmp[1].strip()))

        self._num_texts = len(self.texts)

        # Preprocess sentences
        for i in range(self._num_texts):
            self.texts[i] = self.texts[i].strip()

            # Add <sos> self.__add_sos times
            if self.__add_sos > 0:
                for s in range(self.__add_sos):
                    self.__textA[i] = " ".join(['<SOS>', self.__textA[i]])

            self.__textA[i] = self.__textA[i].strip()
            self.__textA[i] = self.__textA[i].replace(",", "").replace(".", "")
            self.__textA[i] = (
                self.__textA[i].replace("'", "").replace("\?", "").replace("\!", "")
            )

            # Add <eos> self.__add_sos times
            if self.__add_sos > 0:
                for s in range(self.__add_sos):
                    self.__textA[i] = " ".join([self.__textA[i], '<EOS>'])

            self.__textA[i] = self.__textA[i].lower().split(" ")

            if self._max_len < len(self.__textA[i]):
                self._max_len = len(self.__textA[i])

        # Create vocabulary set
        for sentence in self.__textA:
            self.__vocab.update(sentence)

        self.__vocab = sorted(self.__vocab)
        if self.__padding == True:
            """
            The index of "<pad>" must be 0, because the loss_function assumes it.
            """
            self.__vocab.insert(0, '<pad>')

        self._vocab_size = len(self.__vocab)

        for i, w in enumerate(self.__vocab):
            self.word2idx[w] = i

        for w, i in self.word2idx.items():
            self.idx2word[i] = w

        # Create tokenized dataset
        for i in range(len(self.__textA)):
            s = [self.word2idx[s] for s in self.__textA[i]]
            self.sentences.append(s)


    def max_len(self):
        return self._max_len

    def num_texts(self):
        return self._num_texts

    def vocab_size(self):
        return self._vocab_size

    def detokenize(self, tensors):
        sentence = ""
        for i in range(len(tensors)):
            if tensors[i] < self.vocab_size():
                w = self.idx2word[tensors[i]]
                if i == 0: # '<SOS>':
                    sentence = w
                elif i == 1:
                    sentence += ' ' + w.capitalize()
                elif w == ',' or w == '.':
                    sentence += w
                else:
                    if w == '<pad>':
                        # remove '<pad>'
                        continue
                    else:
                        sentence += ' ' + w

        return sentence
