import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet_3d2d import ResNet18 as ResNet18_3d2d
from modules.front_resnet_3d import ResNet18 as ResNet18_3d
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout import Classifier


class ResNet18_3d2d(nn.Module):

    def __init__(self, classes, input_channel=1,in_planes=16, embedding_size=64):
       
        super(ResNet18_3d2d, self).__init__()
        self.front = ResNet18_3d2d(in_planes)
        self.pool = AvgPoolStd()
        self.back = Classifier(classes, in_planes*8*2, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out, embd
    
class ResNet18_3d3d(nn.Module):

    def __init__(self, classes, input_channel=1,in_planes=16, embedding_size=64):
       
        super(ResNet18_3d3d, self).__init__()
        self.front = ResNet18_3d(in_planes)
        self.pool = AvgPoolStd()
        self.back = Classifier(classes, in_planes*8*2, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out, embd = self.back(out)
        return out, embd

        
