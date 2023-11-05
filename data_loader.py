import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


def loader(transform, mode='train',
           batch_size=1, vocab_threshold=None,
           vocab_file='./vocab.pkl',
           start_word='<start>',
           end_word='<end>',
           unk_word='<unk',
           vocab_from_file=False,  # set to false because the file has to be created first
           num_workers=0,  # number of subprocesses to use for data loading
           data_loc='/flickr30k_processed'):

    # to generate vocab mode has to be train
    if vocab_from_file == False:
        assert mode == 'train'

    if mode == 'train':
        images_folder = os.path.join('./flickr30k_processed/images')
        captions_file = os.path.join('./flickr30k_processed/train.csv')

    elif mode == 'test':
        images_folder = os.path.join('./flickr30k_processed/images')
        captions_file = os.path.join('./flickr30k_processed/test.csv')

    dataset = FlickrDataset()


class FlickrDataset(data.Dataset):

    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word,
                 end_word, unk_word, captions_file, vocab_from_file, images_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, captions_file, vocab_from_file)
        self.images_folder = images_folder
