import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pool_atten import AttentivePool
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout import Classifier
from modules.front_resnet import BasicBlock
from modules.mytransformer import make_model

class TransformerAvg(nn.Module):

    def __init__(self, output_dims, N = 2, d_model=128, d_ff=512, h=4, dropout=0.1, embedding_size=16):
       
        super(TransformerAvg, self).__init__()
        self.front = make_model(N = N, d_model = d_model, d_ff = d_ff, h = h, dropout = dropout)
        self.transition1 = BasicBlock(1,8,2)
        self.transition2 = BasicBlock(8,1,2)
        self.back1 = Classifier(output_dims[0], d_model//4, embedding_size)
        self.back2 = Classifier(output_dims[1], d_model//4, embedding_size)
        self.back3 = Classifier(output_dims[2], d_model//4, embedding_size)
        self.back4 = Classifier(output_dims[3], d_model//4, embedding_size)
        self.back5 = Classifier(output_dims[4], d_model//4, embedding_size)
        self.back6 = Classifier(output_dims[5], d_model//4, embedding_size)
        self.back7 = Classifier(output_dims[6], d_model//4, embedding_size)
        self.back8 = Classifier(output_dims[7], d_model//4, embedding_size)
        self.back9 = Classifier(output_dims[8], d_model//4, embedding_size)
        self.back10 = Classifier(output_dims[9], d_model//4, embedding_size)

    def forward(self, x, mask = None):
        # input [B*frames*bins]
        out = self.front(x, mask)
        out = self.transition1(out.unsqueeze(dim=1))
        out = self.transition2(out)
        out = torch.max(out.squeeze(dim=1), dim=1)[0]
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

        return {"out1": out1, "out2": out2, "out3": out3, \
                "out4": out4, "out5": out5, "out6": out6, \
                "out7": out7, "out8": out8, "out9": out9, \
                "out10": out10, "out11": out10, "emb1": embd1, \
                "emb2": embd2, "emb3": embd3, "emb4": embd4, \
                "emb5": embd5, "emb6": embd6, "emb7": embd7, \
                "emb8": embd8, "emb9": embd9, "emb10": embd10, \
                "emb11": embd10}

