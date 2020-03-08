import torch.nn as nn

class Conv(nn.Sequential):
    """
    Pytorch defination for one set of convolution layer
    the building block of all convolution layer used in the network

    ...

    Attributes
    ----------
    in_channels : int
        input channels to the convolution layer set
    out_channels : int
        output channels by the convolution layer set
    kernel_size : int
        size of the convolution kernel
    stride : int
        stride setting for convolution layer
    padding : int
        data padding for convolution layer
    groups : int
        number of groups for group convolution
    """
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

def make_divisible(x, y):
    """
    make number x divisible by number y
    used to satisfy the group convolution shape constraint

    Parameters
    ----------
    x : int
        number x
    y : int
        number y

    Returns
    ------
    x: int
        the new x that is divisible by y
    """
    return int((x // y + 1) * y) if x % y else int(x)
