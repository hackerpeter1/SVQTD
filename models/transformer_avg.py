import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.front_resnet import ResNet18
from modules.pool_atten import AttentivePool
from modules.pool_avgstd import AvgPoolStd
from modules.back_fc_embd_dropout import Classifier
from modules.front_resnet import BasicBlock
from modules.mytransformer import make_model


class TransformerAvg(nn.Module):
    def __init__(self, classes=4, N=3, d_model=128, d_ff=512, h=4, dropout=0.3, embedding_size=16):
        super(TransformerAvg, self).__init__()
        self.front = make_model(N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
        self.transition1 = BasicBlock(2, 32, 2)
        #self.front = make_model(N=1, d_model=d_model // 2, d_ff=d_ff // 2, h=h, dropout=dropout)
        self.transition2 = BasicBlock(32, 32, 2)
        #self.front = make_model(N=1, d_model=d_model // 4, d_ff=d_ff // 4, h=h, dropout=dropout)

        # self.transition = ResNet18(2,16)
        # self.pool = AttentivePool(16)

        self.pool = AvgPoolStd()

        self.back = Classifier(classes, 64, embedding_size)
        # self.back = Classifier(classes, d_model, embedding_size, dropout_rate = dropout)

    def forward(self, x, mask=None):
        # input [B*frames*bins]
        out = self.front(x, mask)
        # --------------------------------------------------------------------------------
        out = torch.cat([out.unsqueeze(dim=1), x.unsqueeze(dim=1)], dim=1)
        # out = self.transition(out)
        out = self.transition1(out)
        out = self.transition2(out)
        out = self.pool(out)
        # --------------------------------------------------------------------------------
        # out = out.unsqueeze(dim=1)
        # out = self.transition(out)
        # out = self.transition1(out)
        # out = self.transition2(out)
        # out = torch.max(out.squeeze(dim=1), dim=1)[0]
        # out = self.pool(out)
        # --------------------------------------------------------------------------------
        # out = self.transition1(out.unsqueeze(dim=1))
        # out = self.front2(out.squeeze(dim=1), mask)
        # out = self.transition2(out.unsqueeze(dim=1))
        # out = self.front3(out.squeeze(dim=1), mask)

        #### for max transformer please modify this out = torch.max(out, dim=1)[0]
        # --------------------------------------------------------------------------------

        out, embd = self.back(out)

        return {"out": out, "embd": embd}
