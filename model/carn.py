"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
Referenced from PCARN-pytorch, https://github.com/nmhkahn/PCARN-pytorch
"""
import torch
import torch.nn as nn
from model import ops

class Group(nn.Module):
    def __init__(self, num_channels, num_blocks, res_scale=1.0):
        super().__init__()

        for nb in range(num_blocks):
            setattr(self,
                "b{}".format(nb+1),
                ops.ResBlock(num_channels, res_scale)
            )
            setattr(self,
                "c{}".format(nb+1),
                nn.Conv2d(num_channels*(nb+2), num_channels, 1, 1, 0)
            )
        self.num_blocks = num_blocks

    def forward(self, x):
        c = out = x
        for nb in range(self.num_blocks):
            unit_b = getattr(self, "b{}".format(nb+1))
            unit_c = getattr(self, "c{}".format(nb+1))

            b = unit_b(out)
            c = torch.cat([c, b], dim=1)
            out = unit_c(c)

        return out


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.sub_mean = ops.MeanShift(255)
        self.add_mean = ops.MeanShift(255, sign=1)

        head = [
            ops.DownBlock(opt.scale),
            nn.Conv2d(3*opt.scale**2, opt.num_channels, 3, 1, 1)
        ]

        # define body module
        for ng in range(opt.num_groups):
            setattr(self,
                "c{}".format(ng+1),
                nn.Conv2d(opt.num_channels*(ng+2), opt.num_channels, 1, 1, 0)
            )
            setattr(self,
                "b{}".format(ng+1),
                Group(opt.num_channels, opt.num_blocks)
            )

        tail = [
            ops.Upsampler(opt.num_channels, opt.scale),
            nn.Conv2d(opt.num_channels, 3, 3, 1, 1)
        ]

        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)

        self.opt = opt

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        c = out = x
        for ng in range(self.opt.num_groups):
            group = getattr(self, "b{}".format(ng+1))
            conv = getattr(self, "c{}".format(ng+1))

            g = group(out)
            c = torch.cat([c, g], dim=1)
            out = conv(c)
        res = out
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
