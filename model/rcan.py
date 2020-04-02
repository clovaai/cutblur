"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
Referenced from EDSR-PyTorch, https://github.com/thstkdgus35/EDSR-PyTorch
"""
import torch.nn as nn
from model import ops

class CALayer(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, num_channels, reduction, res_scale):
        super().__init__()

        body = [
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
        ]
        body.append(CALayer(num_channels, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, num_channels, num_blocks, reduction, res_scale=1.0):
        super().__init__()

        body = list()
        for _ in range(num_blocks):
            body += [RCAB(num_channels, reduction, res_scale)]
        body += [nn.Conv2d(num_channels, num_channels, 3, 1, 1)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.sub_mean = ops.MeanShift(255)
        self.add_mean = ops.MeanShift(255, sign=1)

        head = [
            ops.DownBlock(opt.scale),
            nn.Conv2d(3*opt.scale**2, opt.num_channels, 3, 1, 1)
        ]

        body = list()
        for _ in range(opt.num_groups):
            body += [
                Group(opt.num_channels, opt.num_blocks, opt.reduction, opt.res_scale
            )]
        body += [nn.Conv2d(opt.num_channels, opt.num_channels, 3, 1, 1)]

        tail = [
            ops.Upsampler(opt.num_channels, opt.scale),
            nn.Conv2d(opt.num_channels, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

        self.opt = opt

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
