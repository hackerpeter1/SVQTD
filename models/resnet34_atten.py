import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.back_fc import Classifier
from modules.front_resnet import ResNet34
from modules.pool_atten import AttentivePool


class ResNet34AttenNet(nn.Module):

    def __init__(self, classes, in_planes=16, embedding_size=64):

        super(ResNet34AttenNet, self).__init__()
        self.front = ResNet34(in_planes)
        self.pool = AttentivePool(in_planes * 8)
        self.back = Classifier(classes, in_planes*8, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out, embd

