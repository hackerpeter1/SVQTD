import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze(dim=2)


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze(dim=2)


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


class AttentivePool(nn.Module):

    def __init__(self, feature_size = 128):
        super(AttentivePool, self).__init__()
        self.weight_proj = nn.Parameter(torch.Tensor(feature_size, 1))
        self.weight_W = nn.Parameter(torch.Tensor(feature_size, feature_size))
        self.bias = nn.Parameter(torch.Tensor(feature_size,1))
        self.softmax = nn.Softmax(dim=1)
        self.weight_proj.data.uniform_(-0.1, 0.1)
        self.weight_W.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        out = F.avg_pool2d(x, [x.size()[2], 1], stride=1)
        out = out.squeeze(dim=2)
        out = out.permute(2, 0, 1)
        squish = batch_matmul_bias(out, self.weight_W, self.bias, nonlinearity='tanh')
        attn = batch_matmul(squish, self.weight_proj)
        attn_norm = self.softmax(attn.transpose(1,0))
        attn_vectors = attention_mul(out, attn_norm.transpose(1,0))
        return attn_vectors
    

