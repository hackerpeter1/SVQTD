import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class AvgPool(nn.Module):

    def __init__(self):

        super(AvgPool, self).__init__()

    def forward(self, x):
        out = F.avg_pool2d(x, (x.size()[2], x.size()[3]))
        out = out.view(out.size()[0], -1)
        return out
