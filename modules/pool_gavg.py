import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class GAvgPool(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(GAvgPool, self).__init__()
        self.conv1 = self.conv1x1(in_channels, out_channels, 1)	

    def conv1x1(self, in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x) 
        out = F.avg_pool2d(out, (out.size()[2], out.size()[3]))
        out = out.view(out.size()[0], -1)
        return out
