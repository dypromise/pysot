# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1,
                    padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                  kernel_size=kernel_size, groups=in_channels,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels, kernel_size=1),
    )


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, base_size=8, crop_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.base_size = base_size
        self.crop_size = crop_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 15:
            l = self.base_size // 2
            r = l + self.crop_size
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                out.append(adj_layer(features[i]))
            return out


class AdjustLayerCEM(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors,
                 base_size=8, crop_size=7):
        super(AdjustLayerCEM, self).__init__()
        self.num = len(in_channels)
        self.base_size = base_size
        self.crop_size = crop_size
        self.contextEM = ContextEnhancementModule(
            in_channels, out_channels, scale_factors)

    def forward(self, xs):
        xs_ = xs[-self.num:]
        x = self.contextEM(xs_)
        if x.size(3) < 20:  # shufflenetv2
            l = self.base_size // 2
            r = l + self.crop_size
            x = x[:, :, l:r, l:r]
        return x


class ContextEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factors):
        super(ContextEnhancementModule, self).__init__()
        self.num = len(in_channels)
        for i in range(self.num):
            self.add_module(
                'conv' + str(i + 2),
                nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels[i],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels[0]),
                    nn.ReLU(inplace=True)
                )
            )
        self.scale_factors = scale_factors

    def forward(self, xs):
        out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'conv' + str(i + 2))
            x = adj_layer(xs[i])
            x = nn.functional.interpolate(
                x, scale_factor=self.scale_factors[i], mode='bilinear')
            out.append(x)
        return sum(out)


class AdjustUpsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, adjust=True, scale_factor=2,
                 base_size=8, crop_size=8):
        super(AdjustUpsampleLayer, self).__init__()
        self.adjust = adjust
        if self.adjust:
            self.conv = SeperableConv2d(
                in_channel, out_channel, kernel_size=3, padding=1)
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.adjust:
            x = self.conv(x)
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode='bilinear')
        if x.size(3) < 20:
            l = self.base_size // 2
            r = l + self.crop_size
            x = x[:, :, l:r, l:r]
        return x


class UpsampleAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adjusts, scale_factors,
                 base_size, crop_size):
        super(UpsampleAllLayer, self).__init__()
        self.num = len(in_channels)
        if self.num == 1:
            self.upsample = AdjustUpsampleLayer(
                in_channels[0], out_channels[0], adjusts[0], scale_factors[0],
                base_size, crop_size)
        else:
            for i in range(self.num):
                self.add_module(
                    'upsample' + str(i + 2),
                    AdjustUpsampleLayer(
                        in_channels[i], out_channels[i], adjusts[i],
                        scale_factors[i], base_size, crop_size))

    def forward(self, features):
        features = features[-self.num:]
        if self.num == 1:
            return self.upsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'upsample' + str(i + 2))
                out.append(adj_layer(features[i]))
            return out
