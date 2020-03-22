import torch
import torch.nn as nn

import random
random.seed(41)

from .helpers import Conv, make_divisible


class _DenseLayer(nn.Module):

    """
    Pytorch defination for one dense layer.

    ...

    Attributes
    ----------
    block_number : int
        block number the dense layer belongs to
    in_block_channels : int
        incoming channels to the layer's block
    in_block_transition_channels : int
        incoming transition channels to the layer's block from the previous block
    i_layer : int
        layers sequence in the layer's block
    args : args
        user input perimeter for the network

    Methods
    -------
    forward(x)
        forward function definition for Pytorch
    """

    def __init__(self, block_number, in_block_channels, in_block_transition_channels, i_layer, args):
        super(_DenseLayer, self).__init__()
        self.args = args
        current_block_channels = args.growth[block_number] * i_layer

        '''dense connection index selection'''
        # for the first block, everything is dense connection
        if (block_number == 0):
            self.idx = range(0, in_block_channels+current_block_channels)

        # strating from the second block, previous block gets discounted base on the cross-block connection rate
        else:
            # select the dense connection indexs for previous block base on the cross-block connection rate
            sample_channels = in_block_channels - in_block_transition_channels
            self.idx = random.sample(range(sample_channels), k=make_divisible(sample_channels*args.cross_block_rate, args.group_1x1))

            # add all indexs for all last layer transition and current layers
            self.idx = [*self.idx, *range(sample_channels, in_block_channels+current_block_channels)]

        '''cnn definition'''
        # 1x1 conv i --> b*k
        self.conv_1 = Conv(len(self.idx), args.bottleneck * args.growth[block_number],
                           kernel_size=1, groups=args.group_1x1)
        # 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * args.growth[block_number], args.growth[block_number],
                           kernel_size=3, padding=1, groups=args.group_3x3)


    def forward(self, x):
        x_ = x

        '''convolution on selected x'''
        x = torch.index_select(x, 1, torch.tensor(self.idx).to(self.args.device))
        x = self.conv_1(x)
        x = self.conv_2(x)

        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):

    """
    class defination for one dense block

    ...

    Attributes
    ----------
    block_number : int
        block number the dense layer belongs to
    in_block_channels : int
        incoming channels to the layer's block
    in_block_transition_channels : int
        incoming transition channels to the layer's block from the previous block
    args : args
        user input perimeter for the network
    """

    def __init__(self, block_number, in_block_channels, in_block_transition_channels, args):
        super(_DenseBlock, self).__init__()

        '''construct and add layers in the block'''
        for i_layer in range(args.stages[block_number]):
            layer = _DenseLayer(block_number, in_block_channels, in_block_transition_channels, i_layer, args)
            self.add_module('denselayer_%d' % (i_layer + 1), layer)


class _Transition(nn.Module):

    """
    Pytorch defination for one transition layer, used at the end of each dense block

    ...

    Attributes
    ----------
    in_block_channels : int
        incoming channels to the layer's block
    in_channels : int
        input channels to the layer
    out_channels : int
        input channels by the layer
    args : args
        user input perimeter for the network

    Methods
    -------
    forward(x)
        forward function definition for Pytorch
    """

    def __init__(self, in_block_channels, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        self.args = args
        self.idx = range(in_block_channels, in_block_channels+in_channels)
        self.conv = Conv(in_channels, out_channels,
                         kernel_size=1, groups=args.group_1x1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_ = x

        '''convolution on selected x'''
        x = torch.index_select(x, 1, torch.tensor(self.idx).to(self.args.device))
        x = self.conv(x)

        '''size reduction, average pooling on all x'''
        x = torch.cat([x_, x], 1)
        x = self.pool(x)
        return x
