import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avg import AvgPool
from modules.back_fc_embd import Classifier


class ResNet34AvgNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=16):

        super(ResNet34AvgNet, self).__init__()
        self.front = ResNet34(in_planes)
        self.pool = AvgPool()
        self.back = Classifier(classes, in_planes*8, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return {"out": out, "embd": embd}

        
