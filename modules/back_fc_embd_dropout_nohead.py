import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

class Classifier(nn.Module):

    def __init__(self, num_classes, input_size=16, embedding_size=64,rate=0.5):

        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, embedding_size)
        self.drop = nn.Dropout(rate)
    def forward(self, x):

        embd = self.fc1(x)
        embd = self.drop(embd)

        return embd


