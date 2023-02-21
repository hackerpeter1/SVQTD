import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.front_resnet import ResNet34
from modules.pool_atten import AttentivePool
from modules.back_fc import FourClassifierDropout
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout import Classifier

class ResNet34AvgDropNet(nn.Module):

    def __init__(self, output_dims, input_channel=1, in_planes=16, embedding_size=16):
       
        super(ResNet34AvgDropNet, self).__init__()
        self.front = ResNet34(input_channel,in_planes)
        #self.pool1 = AttentivePool()
        #self.pool2 = AttentivePool(in_planes * 8)
        self.pool = AvgPoolStd()
        self.back1 = Classifier(output_dims[0], in_planes*8*2, embedding_size)
        self.back2 = Classifier(output_dims[1], in_planes*8*2, embedding_size)
        self.back3 = Classifier(output_dims[2], in_planes*8*2, embedding_size)
        self.back4 = Classifier(output_dims[3], in_planes*8*2, embedding_size)
        self.back5 = Classifier(output_dims[4], in_planes*8*2, embedding_size)
        self.back6 = Classifier(output_dims[5], in_planes*8*2, embedding_size)
        self.back7 = Classifier(output_dims[6], in_planes*8*2, embedding_size)
        self.back8 = Classifier(output_dims[7], in_planes*8*2, embedding_size)
        self.back9 = Classifier(output_dims[8], in_planes*8*2, embedding_size)
        self.back10 = Classifier(output_dims[9], in_planes*8*2, embedding_size)
        #self.back11 = Classifier(output_dims[10], in_planes*8*2, embedding_size,dropout_rate=dropout_rate)
        #self.back = FourClassifierDropout(output_dim, in_planes*8, embedding_size)

    def forward(self, x):
        out = self.front(x)
        out = self.pool(out)
        out1, embd1 = self.back1(out)
        out2, embd2 = self.back2(out)
        out3, embd3 = self.back3(out)
        out4, embd4 = self.back4(out)
        out5, embd5 = self.back5(out)
        out6, embd6 = self.back6(out)
        out7, embd7 = self.back7(out)
        out8, embd8 = self.back8(out)
        out9, embd9 = self.back9(out)
        out10, embd10 = self.back10(out)
        #out11, embd11 = self.back11(out)

        return {"out1": out1, "out2": out2, "out3": out3, \
                "out4": out4, "out5": out5, "out6": out6, \
                "out7": out7, "out8": out8, "out9": out9, \
                "out10": out10, "out11": out10, "emb1": embd1, \
                "emb2": embd2, "emb3": embd3, "emb4": embd4, \
                "emb5": embd5, "emb6": embd6, "emb7": embd7, \
                "emb8": embd8, "emb9": embd9, "emb10": embd10, \
                "emb11": embd10}

