import torch
import torch.nn as nn

from random import sample

from .helpers import make_divisible
from .layers import Conv

'''network architecture'''
class _DenseLayer(nn.Module):
    def __init__(self, block_number, in_block_channels, last_transition_channels, i_layer, growth_rate, args):
        super(_DenseLayer, self).__init__()
        current_block_channels = growth_rate * i_layer

        '''dense connection index selection'''
        # for the first block, everything is dense connection
        if (block_number == 0):
            self.idx = range(0, in_block_channels+current_block_channels)

        # strating from the second block, previous block gets discounted base on the cross-block connection rate
        else:
            # select the dense connection indexs for previous block base on the cross-block connection rate
            sample_channels = in_block_channels - last_transition_channels
            self.idx = sample(range(sample_channels), k=make_divisible(sample_channels*args.cross_block_rate, args.group_1x1))

            # add all indexs for all last layer transition and current layers
            self.idx = [*self.idx, *range(sample_channels, in_block_channels+current_block_channels)]

        '''cnn definition'''
        # 1x1 conv i --> b*k
        self.conv_1 = Conv(len(self.idx), args.bottleneck * growth_rate,
                           kernel_size=1, groups=args.group_1x1)
        # 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=args.group_3x3)


    def forward(self, x):
        x_ = x

        # cnn with selected dense connections
        x = torch.index_select(x, 1, torch.tensor(self.idx))
        x = self.conv_1(x)
        x = self.conv_2(x)

        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, block_number, num_layers, in_block_channels, last_transition_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()

        for i_layer in range(num_layers):

            layer = _DenseLayer(block_number, in_block_channels, last_transition_channels, i_layer, growth_rate, args)
            self.add_module('denselayer_%d' % (i_layer + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_block_channels, in_channels, out_channels, args):
        super(_Transition, self).__init__()
        self.idx = range(in_block_channels, in_block_channels+in_channels)
        self.conv = Conv(in_channels, out_channels,
                         kernel_size=1, groups=args.group_1x1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_ = x

        # transition conv
        x = torch.index_select(x, 1, torch.tensor(self.idx))
        x = self.conv(x)

        # avg pooling for all x
        x = torch.cat([x_, x], 1)
        x = self.pool(x)
        return x
