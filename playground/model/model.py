import torch
import torch.nn as nn
import math
from copy import deepcopy

from helpers import _DenseLayer, _FinalConv, Transition

class Model(nn.Module):

    def __init__(self, connections, device):

        super(Model, self).__init__()

        '''variable setup'''

        # parameter config for MNIST
        # 28 -> 14 -> 7
        self.pool_size = 8   # pooling after the final block, 8 is for input of 32*32 and 3 blocks architecture
        self.num_classes = 10

        # variable initialization
        self.features = nn.Sequential()
        self.growth = 4 # initial feature size for the first block

        '''initial convolution layer'''
        self.features.add_module('init_conv', nn.Conv2d(1, self.growth,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1,
                                                        bias=False))

        '''block 1'''
        idx = list(range(4))
        layer = _DenseLayer(idx, device, self.growth)
        self.features.add_module('layer_1', layer)

        if connections[0] == 1:
            idx = list(range(8))
        else:
            idx = list(range(4,8))
        layer = _DenseLayer(idx, device, self.growth)
        self.features.add_module('layer_2', layer)

        '''transition'''
        layer = Transition(12)
        self.features.add_module('transition_1', layer)

        '''block 2'''
        if connections[1] == 1 and connections[4] == 1:
            idx = list(range(12))
        elif connections[1] == 1:
            idx = list(range(4)) + list(range(8,12))
        elif connections[4] == 1:
            idx = list(range(4,12))
        else:
            idx = list(range(8,12))
        layer = _DenseLayer(idx, device, self.growth)
        self.features.add_module('layer_3', layer)

        if connections[2] == 1 and connections[5] == 1 and connections[7] == 1:
            idx = list(range(16))
        elif connections[2] == 1 and connections[5] == 1:
            idx = list(range(8)) + list(range(12,16))
        elif connections[5] == 1 and connections[7] == 1:
            idx = list(range(4,16))
        elif connections[2] == 1 and connections[7] == 1:
            idx = list(range(4)) + list(range(8,16))
        elif connections[2] == 1:
            idx = list(range(4)) + list(range(12,16))
        elif connections[5] == 1:
            idx = list(range(4,8)) + list(range(12,16))
        elif connections[7] == 1:
            idx = list(range(8,16))
        else:
            idx = list(range(12,16))
        layer = _DenseLayer(idx, device, self.growth)
        self.features.add_module('layer_4', layer)

        '''final convolution layer'''
        if connections[3] == 1 and connections[6] == 1 and connections[8] == 1 and connections[9] == 1:
            idx = list(range(20))
        elif connections[6] == 1 and connections[8] == 1 and connections[9] == 1:
            idx = list(range(4,20))
        elif connections[3] == 1 and connections[8] == 1 and connections[9] == 1:
            idx = list(range(4)) + list(range(8,20))
        elif connections[3] == 1 and connections[6] == 1 and connections[9] == 1:
            idx = list(range(8)) + list(range(12,20))
        elif connections[3] == 1 and connections[6] == 1 and connections[8] == 1:
            idx = list(range(12)) + list(range(16,20))
        elif connections[3] == 1 and connections[6] == 1:
            idx = list(range(8)) + list(range(16,20))
        elif connections[3] == 1 and connections[8] == 1:
            idx = list(range(4)) + list(range(8,12)) + list(range(16,20))
        elif connections[3] == 1 and connections[9] == 1:
            idx = list(range(4)) + list(range(12,20))
        elif connections[6] == 1 and connections[8] == 1:
            idx = list(range(4,12)) + list(range(16,20))
        elif connections[6] == 1 and connections[9] == 1:
            idx = list(range(4,8)) + list(range(12,20))
        elif connections[8] == 1 and connections[9] == 1:
            idx = list(range(8,20))
        elif connections[3] == 1:
            idx = list(range(4)) + list(range(16,20))
        elif connections[6] == 1:
            idx = list(range(4,8)) + list(range(16,20))
        elif connections[8] == 1:
            idx = list(range(8,12)) + list(range(16,20))
        elif connections[9] == 1:
            idx = list(range(12,20))
        else:
            idx = list(range(16,20))
        layer = _FinalConv(idx, device, 2*self.growth)
        self.features.add_module('final_conv', layer)

        '''transition'''
        layer = Transition(2*self.growth)
        self.features.add_module('transition_2', layer)

        # print (self.features)

        '''linear layer'''
        self.classifier = nn.Linear(7*7*2*self.growth, self.num_classes)

        '''initialize network'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
