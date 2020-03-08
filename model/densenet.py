import torch
import torch.nn as nn
import math
from copy import deepcopy

from .architectures import _DenseLayer, _DenseBlock, _Transition
from .helpers import make_divisible


class DenseNet(nn.Module):

    """
    define DenseNet model

    ...

    Attributes
    ----------
    args : args
        user input perimeter for the network

    Methods
    -------
    add_block(block_number, in_block_transition_channels)
        add a dense block to the network
    forward(x)
        forward function definition for Pytorch
    """

    def __init__(self, args):

        super(DenseNet, self).__init__()

        '''variable setup'''
        # check and save arges
        assert len(args.stages) == len(args.growth)
        self.args = args

        # parameter config for CIFAR10
        self.init_stride = 1
        self.pool_size = 8   # pooling after the final block, 8 is for input of 32*32 and 3 blocks architecture
        args.num_classes = 10

        # variable initialization
        self.features = nn.Sequential()
        self.num_features = 2 * args.growth[0]  # initial feature size is 2 times growth rate for first block

        '''initial convolution layer'''
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))

        '''dense block'''
        in_block_transition_channels = 0
        for i in range(len(self.args.stages)):
            # dense block i
            in_block_transition_channels = self.add_block(i, in_block_transition_channels)

        '''linear layer'''
        self.classifier = nn.Linear(self.num_features, args.num_classes)

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


    def add_block(self, block_number, in_block_transition_channels):

        """
        generate and add a dense block to the network

        Parameters
        ----------
        block_number : int
            block sequence in the network
        in_block_transition_channels : int
            incoming transition channels from the previous block

        Returns
        ------
        transition_channels: int
            number of transition channels by this block
        """

        '''setup variable'''
        in_block_channels = deepcopy(self.num_features)

        '''convolution block'''
        block = _DenseBlock(
            block_number=block_number,
            in_block_channels=in_block_channels,
            in_block_transition_channels=in_block_transition_channels,
            args=self.args
        )
        self.features.add_module('denseblock_%d' % (block_number + 1), block)
        self.num_features += self.args.stages[block_number] * self.args.growth[block_number]

        transition_channels = 0

        '''transition after block'''
        # transition between blocks
        if block_number + 1 != len(self.args.stages):
            in_channels = self.args.stages[block_number] * self.args.growth[block_number]
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

        '''return variable'''
        self.num_features += transition_channels
        return transition_channels



    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
