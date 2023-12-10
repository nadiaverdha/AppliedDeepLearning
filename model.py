import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        #selects all the layers of resnet model except the last two
        modules = list(resnet.children())[:-2]
        #encapsulates the selected layers in nn.Sequential module
        self.resnet = nn.Sequential(*modules)
        self.fine_tune()

    # fine tuning encoder's last layers if fine_tune is True
    def fine_tune(self,fine_tune = True):
        for param in self.resnet.parameters():
            param.requires_grad = False

        for child in list(self.resnet.children())[5:]:
            for param in child.parameters():
                param.requires_grad = fine_tune
                
    def forward(self,images):
        features = self.resnet(images)
        #tensor of the image features is returned
        features = features.permute(0,2,3,1)
        return features


#soft attention mechanism - BahdanauAttention
class Attention(nn.Module):
    def __init__(self,encoder_dim, decoder_dim, attention_dim):
        super(Attention,self).__init__()
  
        #linear layer to transform encoder's input
        self.encoder_attn = nn.Linear(encoder_dim,attention_dim)
        
        #linear layer to transform decoder's hidden state
        self.decoder_attn = nn.Linear(decoder_dim,attention_dim)
        
        #linear layer to compute attention scores
        self.full_attn = nn.Linear(attention_dim,1)

    def forward(self,encoder_out, decoder_hidden):
        attn1 = self.encoder_attn(encoder_out)   # (batch_size, num_pixels, attention_dim)
        attn2 = self.decoder_attn(decoder_hidden)  # (batch_size, attention_dim)
        attn = self.full_attn(F.relu(attn1+attn2.unsqueeze(1))) # (batch_size, num_pixels, 1)

        #softmax for calculating weights for weighted encoding based on attention
        alpha = F.softmax(attn, dim = 1) # (batch_size, num_pixels,1)
        attn_weighted_encoding = (encoder_out*alpha).sum(dim = 1)  # (batch_size, encoder_dim)
        alpha = alpha = alpha.squeeze(2) # (batch_size, num_pixels)
        return attn_weighted_encoding,alpha

class DecoderRNN(nn.Module):
    def __init__(self,attention_dim, embed_dim,decoder_dim, vocab_size,device, encoder_dim = 2048, dropout = 0.5,):
        super(DecoderRNN,self).__init__()
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim #feature size of decoder's RNN
        self.vocab_size = vocab_size
        self.device = device
        self.encoder_dim = encoder_dim #feature size of encoded images
        self.dropout = dropout

        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        self.embbeding = nn.Embedding(vocab_size,embed_dim)
        self.dropout = nn.Dropout(p = dropout)

        self.decode_step = nn.LSTMCell(embed_dim+encoder_dim,decoder_dim,bias = True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        #initializes layers w/ uniform distribution for easier convergence
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

    def init_hidden_state(self, encoder_out):
        #intializes the hidden and cell states of the LSTM cell based on the mean of the encoder's output
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
        
    def forward(self,encoder_out,encoded_captions,caption_lens):

        batch_size = encoder_out.size(0)

        #flatten image
        encoder_out = encoder_out.view(batch_size,-1,self.encoder_dim)  #encoded image
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)

        h,c = self.init_hidden_state(encoder_out)

        # <end> token will not be included
        decode_lens = (caption_lens -1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lens),self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(self.device)

        # at each time-step generate a new word in the decoder based on the previous word and the attention weighted encoding
        for t in range(max(decode_lens)):
            batch_size_t = sum([l>t for l in decode_lens])

            #attention weighted encodings
            attention_weighted_encoding,alpha = self.attention(encoder_out[:batch_size_t],h[:batch_size_t])

            #sigmoid gating scalar
            gate = F.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))

            #next word prediction
            preds = self.fc(self.dropout(h))

            #save the prediction and alpha for every time step
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas























