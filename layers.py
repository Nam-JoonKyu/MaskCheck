import torch.nn as nn


class DepthWiseConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size, stride,
                                        padding=padding, dilation=dilation,
                                        groups=channels, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.depthwise_conv(x)

class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(PointWiseConv, self).__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise_conv(x)

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, bias=True):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise_conv = DepthWiseConv(in_channels, kernel_size, stride, padding,
                                            padding_mode, dilation, bias)
        self.pointwise_conv = PointWiseConv(in_channels, out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
