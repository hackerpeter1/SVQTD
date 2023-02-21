import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

class Classifier(nn.Module):

    def __init__(self, num_classes, input_size=16, embedding_size=64):

        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):

        embd = self.fc1(x)
		
        out = self.fc2(embd)

        return out, embd


