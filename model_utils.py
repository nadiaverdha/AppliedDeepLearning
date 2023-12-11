import torch
import pandas as pd
# import torchvision
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import csv
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm



def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best,model_file = '/content/drive/MyDrive/AppliedDeepLearning/image_captioning_best.pth'):

    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }
    filename = 'image_captioning_checkpoint_' + str(epoch) + '.pth'
    torch.save(state, filename)
    if is_best:
        print('Saving the best model')
        torch.save(state,model_file)



def train(train_loader,encoder, decoder, criterion, encoder_optimizer,decoder_optimizer,device,grad_clip,alpha_c):
    losses = []
    decoder.train()
    encoder.train()

    for i, (imgs,caps,cap_lens) in enumerate(tqdm(train_loader),len(train_loader)):
      imgs = imgs.to(device)
      caps = caps.to(device)
      cap_lens = cap_lens.to(device)

      imgs = encoder(imgs)
      scores, caps_sorted, decode_lengths, alphas =  decoder(imgs, caps, cap_lens)

      #from <start> to <end>
      targets = caps_sorted[:,1:]

      scores = pack_padded_sequence(scores, decode_lengths, batch_first=True,enforce_sorted=False).data
      targets = pack_padded_sequence(targets, decode_lengths, batch_first=True,enforce_sorted=False).data

      loss = criterion(scores,targets) + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
      #backpropagation
      decoder_optimizer.zero_grad()
      if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
      loss.backward()

      if grad_clip is not None:
          clip_gradient(decoder_optimizer,grad_clip)
          if encoder_optimizer is not None:
            clip_gradient(encoder_optimizer,grad_clip)

      decoder_optimizer.step()
      if encoder_optimizer is not None:
            encoder_optimizer.step()
      losses.append(loss.item())

    return np.mean(losses)

def validate(val_loader,encoder, decoder, criterion,device,alpha_c):

    losses = []
    decoder.eval()
    encoder.eval()
    references = list()  # true captions
    hypotheses = list()  # predicted captions

    with torch.no_grad():
      for i, (imgs,caps,cap_lens,all_caps) in enumerate(tqdm(val_loader)):
          imgs = imgs.to(device)
          caps = caps.to(device)
          cap_lens = cap_lens.to(device)
          all_caps = all_caps
          imgs = encoder(imgs)

          scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, cap_lens)
          targets = caps_sorted[:, 1:]

          scores_copy = scores.clone()
          scores = pack_padded_sequence(scores, decode_lengths, batch_first=True,enforce_sorted=False).data
          targets = pack_padded_sequence(targets, decode_lengths, batch_first=True,enforce_sorted=False).data

          loss = criterion(scores, targets) + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

          losses.append(loss.item())
          for j in range(len(all_caps)):
            img_caps = all_caps[j].tolist()
            img_captions = list(map(lambda caption: [word for word in caption if word != 1 and word !=0],img_caps))
            references.append(img_captions)

          _, preds = torch.max(scores_copy, dim=2)
          preds = preds.tolist()
          temp_preds = list()
          for j, pred in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
          hypotheses.extend(temp_preds)


          assert len(references) == len(hypotheses)

          bleu1 = corpus_bleu(references, hypotheses, weights = (1.0, 0, 0, 0))
          bleu2 = corpus_bleu(references, hypotheses, weights = (0.5, 0.5, 0, 0))
          bleu3 = corpus_bleu(references, hypotheses, weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, 0))
          bleu4 = corpus_bleu(references, hypotheses)
    return np.mean(losses), bleu1,bleu2, bleu3, bleu4
def evaluate_test(test_loader,encoder, decoder, criterion,device,alpha_c):

    losses = []
    decoder.eval()
    encoder.eval()
    references = list()  # true captions
    hypotheses = list()  # predicted captions

    with torch.no_grad():
      for i, (imgs,caps,cap_lens,all_caps) in enumerate(tqdm(test_loader)):
          imgs = imgs.to(device)
          caps = caps.to(device)
          cap_lens = cap_lens.to(device)
          all_caps = all_caps
          imgs = encoder(imgs)

          scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, cap_lens)
          targets = caps_sorted[:, 1:]

          scores_copy = scores.clone()
          scores = pack_padded_sequence(scores, decode_lengths, batch_first=True,enforce_sorted=False).data
          targets = pack_padded_sequence(targets, decode_lengths, batch_first=True,enforce_sorted=False).data

          for j in range(len(all_caps)):
            img_caps = all_caps[j].tolist()
            img_captions = list(map(lambda caption: [word for word in caption if word != 1 and word !=0],img_caps))
            references.append(img_captions)

          _, preds = torch.max(scores_copy, dim=2)
          preds = preds.tolist()
          temp_preds = list()
          for j, pred in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
          hypotheses.extend(temp_preds)



      bleu1 = corpus_bleu(references, hypotheses, weights = (1.0, 0, 0, 0))
      bleu2 = corpus_bleu(references, hypotheses, weights = (0.5, 0.5, 0, 0))
      bleu3 = corpus_bleu(references, hypotheses, weights = (1.0/3.0, 1.0/3.0, 1.0/3.0, 0))
      bleu4 = corpus_bleu(references, hypotheses)
    return bleu1,bleu2, bleu3, bleu4