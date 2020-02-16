import torch
import torch.nn as nn

from .layers import Conv


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args, previous_layers):
        super(_DenseLayer, self).__init__()
        self.previous_layers = previous_layers

        # 1x1 conv i --> b*k
        self.conv_1 = Conv(in_channels, args.bottleneck * growth_rate,
                           kernel_size=1, groups=args.group_1x1)
        # 3x3 conv b*k --> k
        if self.previous_layers:
            self.conv_2 = Conv(args.bottleneck * growth_rate, int(growth_rate-len(self.previous_layers)*8),   # temporary fit for default config
                               kernel_size=3, padding=1, groups=args.group_3x3)
        else:
            self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                               kernel_size=3, padding=1, groups=args.group_3x3)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.growth_rate = growth_rate


    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        self.x = x

        if self.previous_layers:
            previous_xs = [previous_layer.x for previous_layer in self.previous_layers]
            cross_blocks = []
            while previous_xs:
                previous_xs = [self.pool(previous_x) for previous_x in previous_xs]
                cross_blocks.append(previous_xs.pop(-1))

            # for debug
            print ([x.shape for x in cross_blocks])

            return torch.cat([x_, x, *cross_blocks], 1)
        else:
            return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args, all_blocks):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            previous_layers = []
            for block in all_blocks:
                previous_layers.append(block[i])

            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args, previous_layers)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x
