import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout_nohead import Classifier


class ResNet34AvgDropNet(nn.Module):

    def __init__(self, classes, input_channel=1,in_planes=16, embedding_size=64):

        super(ResNet34AvgDropNet, self).__init__()
        self.front = ResNet34(input_channel,in_planes)
        self.pool = AvgPoolStd()
        self.back = Classifier(classes, in_planes*8*2, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        embd = self.back(out)
        return embd

        
