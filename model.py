# The model is going to consist of a encoder (CNN) & decoder (RNN)

import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):

    def __init__(self,image_emb_dim, device):
        super(Encoder,self).__init__()
        self.device = device
        self.image_emb_dim = image_emb_dim

        resnet = models.resnet50(pretrained = True)
        #freezing the parameters
        for param in resnet.parameters():
            param.requires_grad(False)

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



