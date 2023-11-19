import nltk
import pickle
import os.path
from collections import Counter
import regex as re
from nltk.tokenize import RegexpTokenizer
import csv

class Vocabulary():
    def __init__(self,sentence_splitter = None, vocab_file='vocab.txt',
                 captions_file='./flickr30k_processed/train.csv',vocab_size = 5000):
        self.captions_file = captions_file
        self.vocab_file = vocab_file
        # predefined tokens
        self.PADDING_INDEX = 0
        self.SOS = 1
        self.EOS = 2
        self.UNKNOWN_WORD_INDEX = 3

        self.word2index = {}
        self.index2word = {}
        self.counter = Counter()

        self.vocab_size = vocab_size
        self.size = 0

        if sentence_splitter is None:
            word_regex = r'(?:\w+|<\w+>)'
            sentence_splitter = RegexpTokenizer(word_regex).tokenize
        self.splitter = sentence_splitter

    def add_caption(self,caption):
        self.counter.update(self.splitter(caption))

    def build_vocab(self):
        with open(self.captions_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                caption = line.strip().lower().split(',')[1]
                self.add_caption(caption)

        #adding predefined tokens to the vocab
        self.index2word[self.PADDING_INDEX] = '<pad>'
        self.word2index['<pad>'] = self.PADDING_INDEX
        self.index2word[self.SOS] = '<sos>'
        self.word2index['<sos>'] = self.SOS
        self.index2word[self.EOS] = '<eos>'
        self.word2index['<eos>'] = self.EOS
        self.index2word[self.UNKNOWN_WORD_INDEX] = '<unk>'
        self.word2index['<unk>'] = self.UNKNOWN_WORD_INDEX

        #most common words
        words = self.counter.most_common(self.vocab_size - 4)

        for idx,(word,_) in enumerate(words):
            self.word2index[word] = idx + 4
            self.index2word[idx+4] = word
        self.size = len(self.word2index)

    def idx_to_word(self,index):
        try:
             return self.index2word[index]
        except KeyError:
            return self.UNKNOWN_WORD_INDEX


    def word_to_idx(self,word):
        try:
            return self.word2index[word]
        except KeyError:
            return self.UNKNOWN_WORD_INDEX

    def save_vocab(self):
        with open(self.vocab_file, 'a') as file:
            for word in self.word2index.keys():
                line = f'{word} {self.word2index[word]} \n'
                file.write(line)

    def load_vocab(self):
        self.word2index = dict()
        self.index2word = dict()
        with open(self.vocab_file) as file:
            for line in file:
                line = line.strip().split(' ')
                word, index = line[0], line[1]
                self.word2index[word] = int(index)
                self.index2word[int(index)] = word







