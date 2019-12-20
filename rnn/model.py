import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn

class RecurrentNetwork(nn.Module):
    def __init__(self, **params):
        super(RecurrentNetwork, self).__init__()
        ########## YOUR CODE HERE ##########
        # same Embedding layer as above
        self.embedding = nn.Embedding.from_pretrained(params['vecs'].clone().detach().requires_grad_(True), freeze=False)
        # create a recurrent layer (here, a GRU) which outputs a hidden state of 64
        self.gru = nn.GRU(params['embed_dim'], params['hidden_dim'], num_layers=params['num_layers'],
                           batch_first=True, dropout=params['dropout'])
        # and the final projection layer (64 -> 4)
        self.ffnn = nn.Linear(params['hidden_dim'], params['num_labels'])


    # x is a PaddedSequence for an RNN, shape (B, L)
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        embeds = self.embedding(x).float()
        lens = (x != 0).sum(1)
        # pack the sequence instead of padding so that we no longer have a bunch of 0's to feed into our RNN
        # (Embedding layers don't like packed sequences, which is why we started with a padded sequence)
        p_embeds = rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        _, hn = self.gru(p_embeds)
        # take the final hidden state and feed it through the final dense layer
        hns = hn.split(1, dim=0)
        last_hn = hns[-1]
        pred = self.ffnn(last_hn.squeeze(0))
        return pred



class DenseNetwork(nn.Module):
    def __init__(self,embedding,n,dim,pre_train):
        super(DenseNetwork, self).__init__()
        if pre_train == True:
            self.emb = nn.EmbeddingBag(n,dim,mode='sum').from_pretrained(embedding.float(),freeze=False)
        else:
            self.emb = nn.EmbeddingBag(n,dim,mode='sum')
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim, 32)
        self.bn   = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 4)
    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return(x)
