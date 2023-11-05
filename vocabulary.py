import nltk
import pickle
import os.path
from collections import Counter
import regex as re
from nltk.tokenize import RegexpTokenizer
import csv


class Vocabulary(object):
    def __init__(self, vocab_threshold, vocab_file='./vocab.pkl',
                 captions_file='./flickr30k_processed/train.csv',
                 vocab_from_file=False, sentence_splitter=None, start_word="<start>",
                 end_word="<end>", unk_word="<unk>"):

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.captions_file = captions_file
        self.vocab_from_file = vocab_from_file
        self.captions_file = captions_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def get_vocab(self):
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')

        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        counter = Counter()
        with open(self.captions_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                caption = row['caption']
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

                if i % 10000 == 0:
                    print("[%d/%d] Tokenizing captions..." % (i))

        words = [word for word, cnt in counter.items() if cnt >
                 self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
