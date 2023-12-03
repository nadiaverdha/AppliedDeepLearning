import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # freezing the parameters
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self,images):
        features = self.resnet(images)
        # the tensor image has to be of size (batch, size*size, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()
        features = features.permute(0,2,3,1)
        features = features.view(batch,size_1*size_2,feature_maps)


#additive attention as described by Bahdanau Paper
class Attention(nn.Module):
    def __init__(self,num_features, hidden_dim, output_dim = 1):
        super(Attention,self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # linear layers to learn weights for the attention mechanism
        self.W_a = nn.Linear(self.num_features,self.hidden_dim)
        self.U_a = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.v_a = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,features, decoder_hidden):
        #input:
        # features - returned by encoder
        # decoder_hidden - hidden state output from Decoder

        #output:
        # context - context vector with size (1,2048)
        # attent_weight - expresses feature relevance

        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)

        # applying tangent to combine from the two 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        atten_score = self.v_a(atten_tan)
        #multiplying each vector by its softmax score
        atten_weight = F.softmax(atten_score,dim = 1)
        # sum up the vectors - which produce the attention context
        context = torch.sum(atten_weight* features, dim = 1)
        atten_weight = atten_weight.squeeze(dim = 2)

        return context, atten_weight

class DecoderRNN(nn.Module):
    def __init__(self,num_features, embedding_dim,hidden_dim, vocab_size,p = 0.5):
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        #for scaling the inputs to softmax
        self.sample_temp = 0.5

        #turns vectors into a vector of a specified length
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim + num_features , hidden_dim)
        #final output
        self.fc = nn.Linear(hidden_dim,vocab_size)

        #adding attention layer
        self.attention = Attention(num_features,hidden_dim)
        self.drop = nn.Dropout(p = p)

        #initializing hidden state and cell memory using average feature vector
        self.init_h = nn.Linear(num_features,hidden_dim)
        self.init_c = nn.Linear(num_features,hidden_dim)

    #initializes hidden state and cell memory using average feature vector
    def init_hidden(self,features):
        mean_annotations = torch.mean(features,dim = 1)
        #hidden state (short-term memory)
        h0 = self.init_h(mean_annotations)
        #initial cell state (long-term memory)
        c0 = self.init_c(mean_annotations)
        return h0,c0

    def forward(self,captions, features, sample_prob = 0.0):
        # captions - image captions ; features -  features returned by encoder
        #sample_prob - for scheduled sampling

        embed = self.embeddings(captions)
        h,c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        outputs = torch.zeros(batch_size,seq_len,self.vocab_size).to(device)
        atten_weights = torch.zeros(batch_size,seq_len,feature_size).to(device)


        for t in range(seq_len):
            sample_prob = 0.0 if t== 0 else 0.5
            use_sampling = np.random.random() < sample_prob

            if use_sampling == False:
                word_embed = embed[:,t,:]
            context, atten_weight = self.attention(features,h)
            input_concat = torch.cat([word_embed,context],1)
            h,c = self.lstm(input_concat,(h,c))
            h = self.drop(h)
            output = self.fc(h)

            if use_sampling == True:
                scaled_output = output /self.sample_temp
                scoring = F.log_softmax(scaled_output,dim = 1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:,t,:] = output
            atten_weights[:,t,:] = atten_weight

        return outputs,atten_weights
















