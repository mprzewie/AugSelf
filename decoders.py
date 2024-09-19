"""https://github.com/ServiceNow/synbols-benchmarks/blob/master/generation/backbones/biggan_vae.py"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm as sn

import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class MLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, nlayers):
        super().__init__()
        self.nlayers = nlayers
        for i in range(nlayers):
            if i == 0:
                setattr(self, "linear%d" % i, torch.nn.Linear(ni, nhidden, bias=False))
            else:
                setattr(self, "linear%d" % i, torch.nn.Linear(nhidden, nhidden, bias=False))
            setattr(self, "bn%d" % i, torch.nn.BatchNorm1d(nhidden))
        if nlayers == 0:
            nhidden = ni
        self.linear_out = torch.nn.Linear(nhidden, no)

    def forward(self, x):
        for i in range(self.nlayers):
            linear = getattr(self, "linear%d" % i)
            bn = getattr(self, "bn%d" % i)
            x = linear(x)
            x = F.leaky_relu(bn(x), 0.2, True)
        return self.linear_out(x)



class InterpolateResidualGroup(torch.nn.Module):
    def __init__(self, nblocks, ni, no, z_dim, upsample=False):
        super().__init__()
        self.nblocks = nblocks
        for n in range(nblocks):
            if n == 0:
                setattr(self, "block%d" % n, InterpolateResidualBlock(ni, no, z_dim, upsample=upsample))
            else:
                setattr(self, "block%d" % n, InterpolateResidualBlock(no, no, z_dim, upsample=False))

    def forward(self, x, z):
        for n in range(self.nblocks):
            block = getattr(self, "block%d" % n)
            x = block(x, z)
        return x


class ConditionalBatchNorm(torch.nn.Module):
    def __init__(self, no, z_dim):
        super().__init__()
        self.no = no
        self.bn = torch.nn.BatchNorm2d(no, affine=False)
        self.condition = torch.nn.Linear(z_dim, 2 * no)

    def forward(self, x, z):
        cond = self.condition(z).view(-1, 2 * self.no, 1, 1)
        return self.bn(x) * cond[:, :self.no] + cond[:, self.no:]


class InterpolateResidualBlock(torch.nn.Module):
    def __init__(self, ni, no, z_dim, upsample=False):
        super().__init__()
        self.bn0 = ConditionalBatchNorm(ni, z_dim)
        self.conv0 = torch.nn.Conv2d(ni, no, 3, 1, 1, bias=False)
        self.conv0 = sn(self.conv0)
        self.bn1 = ConditionalBatchNorm(no, z_dim)
        self.conv1 = torch.nn.Conv2d(no, no, 3, 1, 1, bias=False)
        self.conv1 = sn(self.conv1)
        self.upsample = upsample
        self.reduce = ni != no
        if self.reduce:
            self.conv_short = sn(torch.nn.Conv2d(ni, no, 1, 1, 0, bias=False))

    def forward(self, x, z):
        if self.upsample:
            shortcut = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x
        x = F.relu(self.bn0(x, z), True)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv0(x)
        x = F.relu(self.bn1(x, z), True)
        x = self.conv1(x)
        if self.reduce:
            x = self.conv_short(shortcut) + x
        else:
            x = x + shortcut
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_dim, width, in_ch, ratio, in_h, in_w, mlp_width, mlp_depth, hierarchical):
        super().__init__()
        self.mlp = MLP(z_dim, in_ch * in_h * in_w, in_ch * mlp_width, mlp_depth)
        self.in_ch = in_ch
        self.in_h = in_h
        self.in_w = in_w
        self.ratio = ratio
        self.width = width
        self.channels = []
        self.hierarchical = hierarchical
        self.z_dim = z_dim


        if ratio == 32:
            self.channels = (np.array([128, 64, 32, 16, 16]) * width).astype(int)
            self.nblocks = [1, 1, 2, 2, 1]
        elif ratio == 16:
            self.channels = (np.array([128, 64, 32, 16]) * width).astype(int)
            self.nblocks = [1, 1, 1, 1]
        elif ratio == 8:
            self.channels = (np.array([64, 32, 16]) * width).astype(int)
            self.nblocks = [1, 1, 1]
        else:
            raise ValueError(ratio)
        if self.hierarchical:
            self.hierarchical_decoding = torch.nn.Linear(self.z_dim, 16 ** 2 * self.channels[-2], bias=True)

        for i, out_ch in enumerate(self.channels):
            if hierarchical and i == len(self.channels) - 1:
                in_ch *= 2
            setattr(self, 'group%d' % i, InterpolateResidualGroup(self.nblocks[i], in_ch, out_ch, z_dim, upsample=True))
            in_ch = out_ch
        self.bn_out = torch.nn.BatchNorm2d(self.channels[-1])
        self.conv_out = torch.nn.Conv2d(self.channels[-1], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        if self.hierarchical:
            z, z_low = z.chunk(2, 1)
            z_features = self.hierarchical_decoding(z_low).view(-1, self.channels[-2], 16, 16)
        z = z.view(z.size(0), -1)
        x = self.mlp(z)
        x = x.view(-1, self.in_ch, self.in_h, self.in_w)
        for i in range(len(self.channels)):
            group = getattr(self, "group%d" % i)
            if self.hierarchical and i == len(self.channels) - 1:
                x = torch.cat([x, z_features], 1)
            x = group(x, z)
        x = F.relu(self.bn_out(x), True)
        return torch.tanh(self.conv_out(x))






def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)

class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768,
                 sbn=True):  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(
            n + 1)]  # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])])

        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f'width={self.width}'

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

def load_decoder(args):
    if args.model == "resnet18":
        z_dim = 512
        img_size = 96
        fm_size = 3
        ratio = 32

    elif args.model == "resnet50":
        z_dim=2048
        img_size = 224
        fm_size = 7
        ratio = 32
    else:
        raise NotImplementedError(args.model)

    args.decoder_fm_size = fm_size
    args.decoder_input_size = z_dim
    return LightDecoder(
        up_sample_ratio=ratio,
        width=z_dim,
        sbn=False, # convert to syncbn will be perfomed in other section of code
    )
    # decoder_args = dict(
    #     z_dim=z_dim,
    #     width=args.dec_width,
    #     in_ch=z_dim,
    #     ratio= img_size // fm_size,
    #     in_h=fm_size, in_w=fm_size,
    #     mlp_width=fm_size, mlp_depth=args.dec_mlp_depth,
    #     hierarchical=args.dec_hierarchical
    # )
    # print(decoder_args)
    # return Decoder(
    #     z_dim=z_dim,
    #     width=args.dec_width,
    #     in_ch=z_dim,
    #     ratio= img_size // fm_size,
    #     in_h=fm_size, in_w=fm_size,
    #     mlp_width=fm_size, mlp_depth=args.dec_mlp_depth,
    #     hierarchical=args.dec_hierarchical
    # )
