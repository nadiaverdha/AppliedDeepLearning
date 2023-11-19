# The model is going to consist of a encoder (CNN) & decoder (RNN)

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self,image_emb_dim, device):
        super(EncoderCNN,self).__init__()
        self.device = device
        self.image_emb_dim = image_emb_dim
        print(f"Encoder:\n \
                       Encoder dimension: {self.image_emb_dim}")

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #freezing the parameters
        for param in resnet.parameters():
            param.requires_grad_(False)

        #removing the classification head of the pretrained model
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        #final classifier
        self.fc = nn.Linear(resnet.fc.in_features,self.image_emb_dim)

    def forward(self,images):
        features = self.resnet(images)
        features = features.reshape(features.size(0),-1).to(self.device)
        features = self.fc(features).to(self.device)
        return features

class DecoderRNN(nn.Module):
    # the decoder takes as input for the LTSM layer the concatenation of features created by the encoding layer
    # and the embedded captions obtained from the embedding layer
    # final classifier - linear layer w/ output dimension of the size of vocab
    def __init__(self, image_emb_dim,word_emb_dim,hidden_dim,num_layers,vocab_size,device):
        super(DecoderRNN,self).__init__()
        self.image_emb_dim = image_emb_dim
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = device

        #the following states represent the memory of the model
        self.hidden_state_0 = nn.Parameter(torch.zeros((self.num_layers,1,self.hidden_dim)))
        self.cell_state_0 = nn.Parameter(torch.zeros((self.num_layers,1,self.hidden_dim)))

        print(f"Decoder:\n \
                       Encoder Size:  {self.image_emb_dim},\n \
                       Embedding Size: {self.word_emb_dim},\n \
                       LSTM Capacity: {self.hidden_dim},\n \
                       Number of layers: {self.num_layers},\n \
                       Vocabulary Size: {self.vocab_size},\n \
                       ")

        self.lstm = nn.LSTM(self.image_emb_dim+self.word_emb_dim, self.hidden_dim,num_layers = self.num_layers,bidirectional=False)

        #fully-connected layer
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim,self.vocab_size),
                                nn.LogSoftmax(dim=2))

    #forward operation of decoder.
    # the input is passed through LSTM and then through the linear layer
    def forward(self,embedded_captions, features,):
        lstm_input = torch.cat((embedded_captions,features),dim = 2)

        output,(hidden,cell) = self.lstm(lstm_input,(hidden,cell))
        output = output.to(self.device)
        output = self.fc(output) # length = 1,batch,vocab_size
        return output,(hidden,cell)


def get_acc(output,target):
    probability = torch.exp(output)
    equality = (target ==probability.max(dim = 1))[1]
    return equality.float().mean()