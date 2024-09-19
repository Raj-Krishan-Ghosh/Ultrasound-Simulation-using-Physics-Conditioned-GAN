import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self, ch, k, p, py, s, dil):
        # 3x3  =>    ch=, k=3, p=1, s=1
        super(SpectralConv2d, self).__init__()
        self.ch = ch

        self.ydim = nn.Conv2d(
            ch, ch, (k, 1), padding=(
                py, 0), stride=(
                s, 1), dilation=(
                dil, 1), bias=False)
        self.xdim = nn.Conv2d(
            ch, ch, (1, k), padding=(
                0, p), stride=(
                1, s), groups=ch, bias=False)
        self.bias = nn.Conv2d(
            ch, ch, (1, 1), padding=(
                0, 0), stride=(
                1, 1), groups=ch, bias=True)

    def forward(self, x):
        # do y x first follwed by the other spectal decompsed filters!

        op = self.ydim(x)
        op = self.xdim(op)
        op = self.bias(op)

        return op


class ResidualBlock(nn.Module):
    def __init__(self, ch, py, dil):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            SpectralConv2d(ch=ch, k=5, p=2, py=py, s=1, dil=dil),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),
            SpectralConv2d(ch=ch, k=5, p=2, py=py, s=1, dil=dil),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return out


class GeneratorModel(nn.Module):
    def __init__(self, in_ch):
        super(GeneratorModel, self).__init__()
        self.down1 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(in_ch, 32, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(32, k=5, p=2, py=2, s=1, dil=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(32, 64, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(64, k=5, p=2, py=2, s=1, dil=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.res1 = ResidualBlock(64, py=2, dil=1)
        self.res2 = ResidualBlock(64, py=2, dil=1)
        self.res3 = ResidualBlock(64, py=2, dil=1)
        self.res4 = ResidualBlock(64, py=2, dil=1)

        self.down2 = nn.Sequential(
            # 1x1 for channel change
            nn.Conv2d(64, 32, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(32, k=5, p=2, py=2, s=1, dil=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # 1x1 for channel change
            nn.Conv2d(32, 1, 1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            # 3x3 spectral conv
            SpectralConv2d(1, k=1, p=0, py=0, s=1, dil=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.down2(x)

        return x


class DiscriminatorModel(nn.Module):
    def __init__(self, in_ch):
        super(DiscriminatorModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 2, 1, padding=0, stride=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=144, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
