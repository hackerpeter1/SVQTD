import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd import Classifier


class ResNet34AvgNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=64):

        super(ResNet34AvgNet, self).__init__()
        self.front = ResNet34(in_planes)
        self.fc1 = nn.Linear(in_planes*8*2*8, embedding_size)
        self.fc2 = nn.Linear(embedding_size, classes)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        out = self.front(x)
        batch_size,channel_size,H,W = out.shape
        #print(out.shape)
        x_mean = out.mean(dim=2).view(batch_size,-1)
        x_std = out.std(dim=2).view(batch_size,-1)
        out = torch.cat([x_mean, x_std], dim=1)
        embd = self.fc1(out)
        embd = self.drop(embd)
        out = self.fc2(embd)
        return out, embd

        
