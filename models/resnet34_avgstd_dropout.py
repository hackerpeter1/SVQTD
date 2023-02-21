import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout import Classifier


class ResNet34AvgDropNet(nn.Module):

    def __init__(self, classes, input_channel=1, in_planes=16, embedding_size=64,dropout_rate=0.5):
       
        super(ResNet34AvgDropNet, self).__init__()
        self.front = ResNet34(input_channel,in_planes)  # [B * 8C * H * W ]
        self.pool = AvgPoolStd()
        self.back = Classifier(classes, in_planes*8*2, embedding_size,dropout_rate=dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return {'out': out, 'embd': embd}

        
