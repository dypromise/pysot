# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


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
    def __init__(self, in_channels, out_channels, base_size=8, crop_size=7):
        super(AdjustLayerCEM, self).__init__()
        self.base_size = base_size
        self.crop_size = crop_size
        self.contextEM = ContextEnhancementModule(in_channels, out_channels)

    def forward(self, xs):
        x2, x3 = xs
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x = self.contextEM(x2, x3)
        if x.size(3) < 20:  # shufflenetv2
            l = self.base_size // 2
            r = l + self.crop_size
            x = x[:, :, l:r, l:r]
        return x


class ContextEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextEnhancementModule, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels[0], out_channels[0],
        #               kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels[0]),
        #     nn.ReLU(inplace=True),
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0],
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1],
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True),
        )
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x2, x3):
        # x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = nn.functional.interpolate(x2, scale_factor=2)
        x3 = self.conv3(x3)
        x3 = nn.functional.interpolate(x3, scale_factor=4)
        return x2 + x3
