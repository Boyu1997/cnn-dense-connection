import torch
import torch.nn as nn

from .layers import Conv


class _DenseLayer(nn.Module):
    def __init__(self, in_block_channels, i_layer, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.in_channels = in_block_channels + growth_rate * i_layer
        self.out_channels = in_block_channels + growth_rate * (i_layer + 1)

        # 1x1 conv i --> b*k
        self.conv_1 = Conv(self.in_channels, args.bottleneck * growth_rate,
                           kernel_size=1, groups=args.group_1x1)
        # 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=args.group_3x3)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x_ = x
        print (x.size())

        # torch.index_select(x, 0, torch.tensor(idx))
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_block_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()

        for i_layer in range(num_layers):

            layer = _DenseLayer(in_block_channels, i_layer, growth_rate, args)
            self.add_module('denselayer_%d' % (i_layer + 1), layer)


class _Transition(nn.Module):
    def __init__(self, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x
