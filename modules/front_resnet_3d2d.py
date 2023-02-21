'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.bn2(self.conv2(out))
        #out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.bn2(self.conv2(out))
        #out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(1, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.layer1 = self._make_layer(BasicBlock3d, in_planes, num_blocks[0], stride=1)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=(6,1,1), stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        self.layer2 = self._make_layer(BasicBlock2d, in_planes*2, num_blocks[1], stride=2)
        #self.layer2 = self._make_layer(BasicBlock3d, in_planes*2, num_blocks[1], stride=2)
        #self.conv2 = nn.Conv3d(in_planes*2, in_planes*2, kernel_size=(3,1,1), stride=1, bias=False)
        #self.bn2 = nn.BatchNorm3d(in_planes*2)
        self.layer3 = self._make_layer(BasicBlock2d, in_planes*4, num_blocks[2], stride=2)
        #self.layer3 = self._make_layer(BasicBlock3d, in_planes*4, num_blocks[2], stride=2)
        #elf.conv2 = nn.Conv3d(in_planes*4, in_planes*4, kernel_size=(2,1,1), stride=1, bias=False)
        #self.bn2 = nn.BatchNorm3d(in_planes*4)
        self.layer4 = self._make_layer(BasicBlock2d, in_planes*8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = F.relu(self.bn2(self.conv2(out)))[:,:,0,:,:]
        out = self.layer2(out)
        #out = F.relu(self.bn2(self.conv2(out)))[:,:,0,:,:]
        out = self.layer3(out)
        #out = F.relu(self.bn2(self.conv2(out)))[:,:,0,:,:]
        out = self.layer4(out)
        return out


def ResNet18(in_planes):
    return ResNet(in_planes, [2,2,2,2])

