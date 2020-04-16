# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

class SentimentGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, bi=False, layers=1):
        super(SentimentGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bi, num_layers=layers).cuda()
        
        factor = int(2 + (int(bi) * 2))
        # self.fcX = nn.Linear(hidden_size*factor, 200).cuda()
        # self.fc1 = nn.Linear(200, 70).cuda()
        # self.fc2 = nn.Linear(70, 10).cuda()
        # self.fc3 = nn.Linear(10, num_classes).cuda()

        self.fc = nn.Linear(hidden_size*factor, num_classes).cuda()
    
    def forward(self, x):
        # Convert x to one hot
        ident = torch.eye(self.input_size)
        x = ident[x].cuda()

        # Set an initial hidden state
        h0 = torch.zeros(2, x.shape[0], self.hidden_size).cuda()
       
        # Forward propagate the GRU 
        out, _ = self.rnn(x, h0)

        # Get the max and mean vector and concatenate them        
        out = torch.cat([torch.max(out, dim=1)[0], 
                        torch.mean(out, dim=1)], dim=1).cuda() 
       
        # out = self.fcX(out).cuda()
        # out = self.fc1(F.relu(out).cuda()).cuda()
        # out = self.fc2(F.relu(out).cuda()).cuda()
        # out = self.fc3(F.relu(out).cuda()).cuda()

        out = self.fc(out).cuda()

        return out

def convert_to_stoi(vocab, string):

    stoi = []
    for s in string:
        # If character is in vocab, append its index
        # else, append the index of <unk>
        if s in vocab:
            stoi.append(vocab[s])
        else:
            stoi.append(vocab['<unk>'])
    
    return stoi