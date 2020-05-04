import torch
import torch.nn as nn

def id_to_connections(id):
    binary = '{0:010b}'.format(id)
    connections = []
    for char in binary:
        connections.append(int(char))
    return connections


class MeanValueManager():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.mean = 0

    def update(self, value):
        if isinstance(value, list):
            self.count += len(value)
            self.sum += sum(value)
        else:
            self.count += 1
            self.sum += value
        self.mean = self.sum / self.count


def get_parameter_count(model):
    perimeter_count = 0
    for layer_perimeter in list(model.parameters()):
        perimeter_count_in_layer = 1
        for count in list(layer_perimeter.size()):
            perimeter_count_in_layer *= count
        perimeter_count += perimeter_count_in_layer
    return perimeter_count


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super(Conv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


class Transition(nn.Sequential):
    def __init__(self, in_channels):
        super(Transition, self).__init__()
        self.add_module('norm_last', nn.BatchNorm2d(in_channels))
        self.add_module('relu_last', nn.ReLU(inplace=True))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseLayer(nn.Module):
    def __init__(self, idx, device, growth):
        super(_DenseLayer, self).__init__()

        self.idx = idx
        self.device = device
        self.growth = growth
        self.bottleneck = 2

        '''cnn definition'''
        # 1x1 conv i --> b*k
        self.conv_1 = Conv(len(self.idx), self.bottleneck*self.growth,
                           kernel_size=1, groups=4)
        # 3x3 conv b*k --> k
        self.conv_2 = Conv(self.bottleneck*self.growth, self.growth,
                           kernel_size=3, padding=1, groups=4)


    def forward(self, x):
        x_ = x

        '''convolution on selected x'''
        x = torch.index_select(x, 1, torch.tensor(self.idx).to(self.device))
        x = self.conv_1(x)
        x = self.conv_2(x)

        return torch.cat([x_, x], 1)


class _FinalConv(nn.Module):
    def __init__(self, idx, device, out_channels):
        super(_FinalConv, self).__init__()

        self.idx = idx
        self.device = device
        self.out_channels = out_channels

        '''final conv'''
        self.final_conv = Conv(len(self.idx), self.out_channels,
                               kernel_size=3, padding=1, groups=4)

    def forward(self, x):
        '''convolution on selected x'''
        x = torch.index_select(x, 1, torch.tensor(self.idx).to(self.device))
        return self.final_conv(x)
