import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_atten import AttentivePool
from modules.back_fc_embd_dropout import Classifier


class ResNet34AvgDropNet(nn.Module):

    def __init__(self, classes, input_channel=1, in_planes=16, embedding_size=64):
       
        super(ResNet34AvgDropNet, self).__init__()
        self.front = ResNet34(input_channel,in_planes)
        self.pool = AttentivePool(in_planes * 8)
        self.back = Classifier(classes, in_planes*8, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out, embd

        
