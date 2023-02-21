import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self,classes, input_size=16, embedding_size=64,dropout_rate=0.5):

        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, embedding_size)
        self.drop = nn.Dropout(dropout_rate)
        print('Dropout rate',dropout_rate)
        self.fc2 = nn.Linear(embedding_size, classes)

    def forward(self, x):
        embd = self.fc1(x)
        embd = self.drop(embd)
        out = self.fc2(embd)
        return out, embd


