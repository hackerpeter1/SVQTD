import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet101
from modules.pool_avg import AvgPool
from modules.back_fc import Classifier


class ResNet101AvgNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=64):

        super(ResNet101AvgNet, self).__init__()
        self.front = ResNet101(in_planes)
        self.pool = AvgPool()
        self.back = Classifier(classes, in_planes*8*4, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out = self.back(out)
        return out

        
