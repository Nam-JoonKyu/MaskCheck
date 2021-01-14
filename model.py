import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from prettytable import PrettyTable

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, separable, final_activation=True):
        super(DoubleConv, self).__init__()

        self.final_activation = final_activation

        if separable is True:
            self.layers = nn.Sequential(
                DepthWiseSeparableConv(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DepthWiseSeparableConv(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.layers(x)

        if self.final_activation:
            x = F.relu(x, inplace=True)

        return x


class MaskNet(nn.Module):

    def __init__(self, num_channels, separable=True):
        super(MaskNet, self).__init__()

        assert num_channels[0] == 3

        self.separable = separable
        self.layer_list = nn.ModuleList()

        for i in range(1, len(num_channels)-1):
            self.layer_list.append(DoubleConv(num_channels[i-1], num_channels[i], separable))

        self.layer_list.append(DoubleConv(num_channels[-2], num_channels[-1], separable, False))


        self.fc1 = nn.Linear(num_channels[-1]* (128 // 2**(len(num_channels)-1))**2, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x

    def summary(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


if __name__ == '__main__':
    x=torch.randn(1, 3, 128,128)
    num_channels=[3, 64, 128, 256, 512, 512]
    model=MaskNet(num_channels)
    y=model(x)