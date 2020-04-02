"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
import math
import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range, sign=-1,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, num_channels, res_scale=1.0):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, num_channels, scale):
        m = list()
        if (scale & (scale-1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_channels, 4*num_channels, 3, 1, 1)]
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m += [nn.Conv2d(num_channels, 9*num_channels, 3, 1, 1)]
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super().__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale**2), h//self.scale, w//self.scale)
        return x
