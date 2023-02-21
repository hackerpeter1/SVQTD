import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
    
class AvgPoolStd(nn.Module):
    def __init__(self):
        super(AvgPoolStd, self).__init__()

    def forward(self, x):
        out = x.view(x.size()[0], x.size()[1], -1)
        x_mean = out.mean(dim=2)
        x_std = out.std(dim=2)

        out = torch.cat([x_mean, x_std], dim=1)

        return out
