"""
Unet++ for Planet-A Competition
Modified below github repository
Implemented
1. Inside Block Attention
2. Ensemble Channel Attention (ECA)
3. Residual Connection
4. DropOut
"""
# https://github.com/4uiiurz1/pytorch-nested-unet

import torch
from torch import nn


class CA_Layer(nn.Module):
    """
    Channel Attention (CA) Layer

    Args:
        channel (int)   : Number of channels in the input feature map
        reduction (int) : Reduction ratio. Default: 4.

    Returns:
        out (tensor)    : Channel attention value multiplied with the input feature map
    """

    def __init__(self, channel, reduction=4):
        super(CA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                      kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Up(nn.Module):
    """
    Up Convolutional Block

    Args:
        in_ch (int) : Number of channels in the input feature map

    Returns:
        out (tensor) : Up Convolutional Block
    """

    def __init__(self, in_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class VGGBlock(nn.Module):
    """
    Basic Convolutional Block for the network

    Args:
        in_channels (int) : Number of channels in the input feature map
        middle_channels (int) : Number of channels in the middle feature map
        out_channels (int) : Number of channels in the output feature map
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = CA_Layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.ca(x)
        output = self.relu(x + identity)
        return output


class ChannelAttention(nn.Module):
    """
    Self-Attention Block at the last of the model

    Args:
        in_channels (int) : Number of channels in the input feature map
        ratio (int) : Reduction ratio. Default: 16.

    Returns:
        out (tensor) : output of the Self-Attention Block
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class NestedUNet(nn.Module):
    """
    Nested U-Net model (U-Net++)

    Args:
        num_classes (int) : Number of classes in the dataset
        input_channels (int) : Number of channels in the input feature map
        dropout (float) : Dropout rate. Default: 0.2

    Returns:
        out (tensor) : output of the model
    """

    def __init__(self, num_classes, input_channels=16, dropout=0.2):
        super().__init__()

        init = 16
        nb_filter = [init * 2, init * 4, init * 8, init * 16, init * 32]

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.do0_0   = nn.Dropout2d(p=dropout)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.do1_0   = nn.Dropout2d(p=dropout)
        self.up1_0   = Up(nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.do2_0   = nn.Dropout2d(p=dropout)
        self.up2_0   = Up(nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.do3_0   = nn.Dropout2d(p=dropout)
        self.up3_0   = Up(nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.do4_0   = nn.Dropout2d(p=dropout)
        self.up4_0   = Up(nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.do0_1   = nn.Dropout2d(p=dropout)
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.do1_1   = nn.Dropout2d(p=dropout)
        self.up1_1   = Up(nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.do2_1   = nn.Dropout2d(p=dropout)
        self.up2_1   = Up(nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.do3_1   = nn.Dropout2d(p=dropout)
        self.up3_1   = Up(nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.do0_2   = nn.Dropout2d(p=dropout)
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.do1_2   = nn.Dropout2d(p=dropout)
        self.up1_2   = Up(nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.do2_2   = nn.Dropout2d(p=dropout)
        self.up2_2   = Up(nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.do0_3   = nn.Dropout2d(p=dropout)
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.do1_3   = nn.Dropout2d(p=dropout)
        self.up1_3   = Up(nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.do0_4   = nn.Dropout2d(p=dropout)

        self.ca = ChannelAttention(nb_filter[0] * 4)
        self.ca1 = ChannelAttention(nb_filter[0], ratio=16 // 4)

        self.final = nn.Conv2d(nb_filter[0] * 4, num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.do0_0(self.conv0_0(input))
        x1_0 = self.do1_0(self.conv1_0(self.pool(x0_0)))
        x0_1 = self.do0_1(self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1)))

        x2_0 = self.do2_0(self.conv2_0(self.pool(x1_0)))
        x1_1 = self.do1_1(self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1)))
        x0_2 = self.do0_2(self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1)))

        x3_0 = self.do3_0(self.conv3_0(self.pool(x2_0)))
        x2_1 = self.do2_1(self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1)))
        x1_2 = self.do1_2(self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1)))
        x0_3 = self.do0_3(self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1)))

        x4_0 = self.do4_0(self.conv4_0(self.pool(x3_0)))
        x3_1 = self.do3_1(self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1)))
        x2_2 = self.do2_2(self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1)))
        x1_3 = self.do1_3(self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1)))
        x0_4 = self.do0_4(self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1)))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        output = self.final(out)

        return output
