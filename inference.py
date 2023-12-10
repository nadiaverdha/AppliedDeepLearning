import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from loader import FlickrDataset,preprocessing_transforms,get_data_loader,denormalize,Padding
import matplotlib.pyplot as plt

def beam_search(encoder, decoder, image_path, vocab,vocab_size, device,beam_size=5,):

    #how many possibilities will be considered
    k = beam_size
    #preprocessing image
    img = np.array(Image.open(image_path).convert('RGB'))
    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (256, 256))

    transform = v2.Compose([v2.ToTensor(),v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = transform(img)

    #encoding the image
    encoder_out = encoder(img.unsqueeze(0).to(device))
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    #flattens the encoded representation of image
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    #for storing top k previous words at each step
    top_k_prev_words = torch.tensor([[vocab.word_to_idx("<SOS>")]] * k, dtype=torch.long).to(device)

    #for storing top k sequences
    top_k_seqs = top_k_prev_words

    #for storing top k sequences' scores
    top_k_scores = torch.zeros(k, 1).to(device)

    #for storing top k sequences' alphas
    top_k_seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    #lists for storing completed sequences, alphas and scores
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    #start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(top_k_prev_words).squeeze(1)

        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        gate = F.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding

        h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c) )

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        #adding scores to the previous ones
        scores = top_k_scores.expand_as(scores) + scores

        #same score for the first step
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            #unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        #converting unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        # add new words to sequences, alphas
        top_k_seqs = torch.cat([top_k_seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        top_k_seqs_alpha = torch.cat([top_k_seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)


        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 2]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        #setting aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(top_k_seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        #proceed with incomplete sequences
        if k == 0:
            break


        top_k_seqs = top_k_seqs[incomplete_inds]
        top_k_seqs_alpha = top_k_seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        top_k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        #in case the loop goes on for too long
        if step > 50:
            break
        step += 1

    #selecting sequence with max score
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    caption = [vocab.idx_to_word(int(ind)) for ind in seq if ind!=3 and ind!=2  ]
    caption = ' '.join(word for word in caption)

    image = denormalize(img)
    plt.imshow(image)
    return caption

