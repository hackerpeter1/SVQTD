import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random



class Classifier(nn.Module):

    def __init__(self, num_classes, input_size=16, embedding_size=64):

        super(Classifier, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)

    def forward(self, x):

        embd = self.relu(self.bn(self.fc1(x)))
        out = self.fc2(embd)

        return out, embd
'''
class FourClassifier(nn.Module):

    def __init__(self, output_dim , input_size = 16, embedding_size=512):
        
        super(FourClassifier, self).__init__()

        self.activation = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(embedding_size)
        self.norm2 = nn.BatchNorm1d(embedding_size)
        self.norm3 = nn.BatchNorm1d(embedding_size)
        self.norm4 = nn.BatchNorm1d(embedding_size)

        self.fc0 = nn.Linear(input_size, embedding_size)
        ### classifier1
        self.fc1_1 = nn.Linear(embedding_size, embedding_size)
        self.fc1_2 = nn.Linear(embedding_size, output_dim[0])

        ### classifier2
        self.fc2_1 = nn.Linear(embedding_size, embedding_size)
        self.fc2_2 = nn.Linear(embedding_size, output_dim[1])

        ### classifier3 
        self.fc3_1 = nn.Linear(embedding_size, embedding_size)
        self.fc3_2 = nn.Linear(embedding_size, output_dim[2])

        ### classifier4
        self.fc4_1 = nn.Linear(embedding_size, embedding_size)
        self.fc4_2 = nn.Linear(embedding_size, output_dim[3])

    def forward(self, x):
        embd0 = self.fc0(x)        
        embd1 = self.norm1(self.activation(self.fc1_1(embd0)))
        out1 = F.softmax(self.fc1_2(embd1), dim = 1)

        embd2 = self.norm2(self.activation(self.fc2_1(embd0)))
        out2 = F.softmax(self.fc2_2(embd2), dim = 1)

        embd3 = self.norm3(self.activation(self.fc3_1(embd0)))
        out3 = F.softmax(self.fc3_2(embd3), dim = 1)

        embd4 = self.norm4(self.activation(self.fc4_1(embd0)))
        out4 = F.softmax(self.fc4_2(embd4), dim = 1)
        
        return {"out1": out1, "out2": out2, "out3": out3, "out4":out4, \
                "emb0": emb0, "emb1": embd1, "emb2": embd2, "emb3": embd3, "emb4":embd4}
'''
class FourClassifierDropout(nn.Module):
    def __init__(self, output_dim , input_size = 16, embedding_size=512, dropout=0.5):
        
        super(FourClassifierDropout, self).__init__()

        self.activation = nn.PReLU()
        self.norm0 = nn.BatchNorm1d(embedding_size)
        self.norm1 = nn.BatchNorm1d(embedding_size)
        self.norm2 = nn.BatchNorm1d(embedding_size)
        self.norm3 = nn.BatchNorm1d(embedding_size)
        self.norm4 = nn.BatchNorm1d(embedding_size)

        self.fc0 = nn.Linear(input_size, embedding_size)
        ### classifier1
        self.fc1_1 = nn.Linear(embedding_size, embedding_size)
        self.dropout_fc1_1 = torch.nn.Dropout(p=dropout)
        self.fc1_2 = nn.Linear(embedding_size, output_dim[0])

        ### classifier2
        self.fc2_1 = nn.Linear(embedding_size, embedding_size)
        self.dropout_fc2_1 = torch.nn.Dropout(p=dropout)
        self.fc2_2 = nn.Linear(embedding_size, output_dim[1])

        ### classifier3 
        self.fc3_1 = nn.Linear(embedding_size, embedding_size)
        self.dropout_fc3_1 = torch.nn.Dropout(p=dropout)
        self.fc3_2 = nn.Linear(embedding_size, output_dim[2])

        ### classifier4
        self.fc4_1 = nn.Linear(embedding_size, embedding_size)
        self.dropout_fc4_1 = torch.nn.Dropout(p=dropout)
        self.fc4_2 = nn.Linear(embedding_size, output_dim[3])

    def forward(self, x):
        embd0 = self.fc0(x)
        x = self.norm0(self.activation(embd0))
        
        embd1 = self.fc1_1(self.dropout_fc1_1(x))
        out1 = F.softmax(self.fc1_2(self.norm1(self.activation(embd1))), dim = 1)

        embd2 = self.fc2_1(self.dropout_fc2_1(x))
        out2 = F.softmax(self.fc2_2(self.norm2(self.activation(embd2))), dim = 1)

        embd3 = self.fc3_1(self.dropout_fc3_1(x))
        out3 = F.softmax(self.fc3_2(self.norm3(self.activation(embd3))), dim = 1)

        embd4 = self.fc4_1(self.dropout_fc4_1(x))
        out4 = F.softmax(self.fc4_2(self.norm4(self.activation(embd4))), dim = 1)
        
        return {"out1": out1, "out2": out2, "out3": out3, "out4":out4, \
                "emb0": embd0, "emb1": embd1, "emb2": embd2, "emb3": embd3, "emb4":embd4}

       
