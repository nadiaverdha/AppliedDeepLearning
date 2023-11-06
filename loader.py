import nltk
import os

import pandas as pd
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import csv
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def preprocessing_transforms():
    return v2.Compose([
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def denormalize(image):
    inv_normalize = v2.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()


class FlickrDataset(data.Dataset):
    def __init__(self,captions_file, transform ,vocab,images_folder):

        self.images_folder = images_folder
        self.transform = transform
        self.captions_file = captions_file
        self.samples = []
        with open(captions_file, 'r',encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i==0:
                    continue

                sample = line.strip().lower().split(',')
                image_id = sample[2].split('/')[1]
                caption = sample[1]
                caption = '<sos>' + caption +'<eos>'

                words = vocab.splitter(caption)
                word_ids = [vocab.word_to_idx(word) for word in words]

                sample = {'image_id':image_id, 'caption':torch.LongTensor(word_ids)}

                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id =  os.path.join(self.images_folder, sample['image_id'])

        try:
            image = Image.open(img_id).convert('RGB')
        except FileNotFoundError:
            print(f'Could not find image {img_id}')
            image = Image.new('RGB',(256,256))

        if self.transform:
            image = self.transform(image)
        return image, sample['caption']


def get_data_loader(dataset,batch_size,num_workers):
        return DataLoader(dataset=dataset, batch_size= batch_size,
                          num_workers=num_workers,pin_memory=True, shuffle=True,)





