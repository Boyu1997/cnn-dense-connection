import torch
import torch.nn as nn
import math
from copy import deepcopy

from .architectures import _DenseLayer, _DenseBlock, _Transition
from .helpers import make_divisible


class DenseNet(nn.Module):
    def __init__(self, args):

        super(DenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth

        assert len(self.stages) == len(self.growth)
        self.args = args

        # model config for CIFAR10
        self.init_stride = 1
        self.pool_size = 8
        args.num_classes = 10

        self.features = nn.Sequential()
        self.num_features = 2 * self.growth[0]

        # Initial Conv layer (224*224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))

        last_transition_channels = 0
        for i in range(len(self.stages)):
            ### Dense-block i
            last_transition_channels = self.add_block(i, last_transition_channels)

        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def add_block(self, block_number, last_transition_channels):

        in_block_channels = deepcopy(self.num_features)

        block = _DenseBlock(
            block_number=block_number,
            num_layers=self.stages[block_number],
            in_block_channels=in_block_channels,
            last_transition_channels=last_transition_channels,
            growth_rate=self.growth[block_number],
            args=self.args
        )
        self.features.add_module('denseblock_%d' % (block_number + 1), block)
        self.num_features += self.stages[block_number] * self.growth[block_number]

        transition_channels = 0

        # transition between blocks
        if block_number + 1 != len(self.stages):
            in_channels = self.stages[block_number] * self.growth[block_number]
            out_channels = make_divisible(math.ceil(in_channels * self.args.reduction), self.args.group_1x1)
            trans = _Transition(in_block_channels=in_block_channels,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                args=self.args)
            self.features.add_module('transition_%d' % (block_number + 1), trans)
            transition_channels = out_channels

        # activation function for the last block the last block
        else:
            self.features.add_module('norm_last', nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last', nn.ReLU(inplace=True))
            self.features.add_module('pool_last', nn.AvgPool2d(self.pool_size))

        self.num_features += transition_channels
        return transition_channels



    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
