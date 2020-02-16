import torch
import torch.nn as nn
import math

from .architectures import _DenseLayer, _DenseBlock, _Transition


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

        all_blocks = []
        for i in range(len(self.stages)):
            ### Dense-block i
            block = self.add_block(i, all_blocks)
            all_blocks.append(block)

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


    def add_block(self, i, all_blocks):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
            all_blocks=all_blocks
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))
        return block

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
