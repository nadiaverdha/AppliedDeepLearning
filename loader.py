import nltk
import os

import pandas as pd
import torch
import torch.utils.data as data
from vocab import Vocabulary
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import csv
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def preprocessing_transforms():
    # basically standard values found on pytorch docu
    return v2.Compose([
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def denormalize(image):
    inv_normalize = v2.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()


class FlickrDataset(data.Dataset):
    def __init__(self, captions_file, transform, vocab, images_folder, captions_per_image=5, train_mode=True):
        self.images_folder = images_folder
        self.transform = transform
        self.captions_file = captions_file
        self.captions_per_image = captions_per_image
        self.samples = []
        self.all_captions = {}
        self.train_mode = train_mode
        with open(captions_file, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                sample = line.strip().lower().split(',')
                image_id = sample[2].split('/')[1]
                caption = sample[1]
                caption = '<sos>' + caption + '<eos>'
                words = vocab.splitter(caption)
                word_ids = [vocab.word_to_idx(word) for word in words]
                sample = {'image_id': image_id, 'caption': torch.LongTensor(word_ids)}
                self.samples.append(sample)
                if image_id not in self.all_captions:
                    self.all_captions[image_id] = [torch.LongTensor(word_ids)]
                else:
                    self.all_captions[image_id].append(torch.LongTensor(word_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = os.path.join(self.images_folder, sample['image_id'])
        try:
            image = Image.open(img_id).convert('RGB')
        except FileNotFoundError:
            print(f'Could not find image {img_id}')
            image = Image.new('RGB', (256, 256))
        if self.transform:
            image = self.transform(image)
        if self.train_mode:
            return image, sample['caption']
        else:
            return image, sample['caption'], self.all_captions[sample['image_id']]


# by introducing padding I will make sure that all my captions have the same length
class Padding:
    def __init__(self, pad_idx, batch_first=True, train_mode=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.train_mode = train_mode

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        lengths = []
        captions = [item[1] for item in batch]

        for cap in captions:
            lengths.append(len(cap))
        # lengths = [len(cap) for cap in captions]

        captions = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)
        if self.train_mode:
            return imgs, captions, torch.LongTensor(lengths)
        else:
            all_caps = [item[2] for item in batch]
            all_caps = [pad_sequence(item, batch_first=self.batch_first, padding_value=self.pad_idx) for item in
                        all_caps]

            return imgs, captions, torch.LongTensor(lengths), all_caps


def get_data_loader(dataset, batch_size=32, pad_index=0):
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      num_workers=3, pin_memory=True, shuffle=True,
                      collate_fn=Padding(pad_idx=pad_index, batch_first=True, train_mode=True))
