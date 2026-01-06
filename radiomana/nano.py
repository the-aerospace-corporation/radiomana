#!/usr/bin/env python3

import torch
from einops.layers.torch import Rearrange, Reduce
from timm.layers import DropPath
from timm.models.fastvit import (
    MobileOneBlock,
    PatchEmbed,
    ReparamLargeKernelConv,
)
from torch import nn

from .models import ModelBaseClass


class NanoGRU(ModelBaseClass):
    """How small is really possible?"""

    def __init__(self, hidden_size=18, num_classes=9):
        super().__init__(num_classes=num_classes)

        self.channels = [4, 8, 16]
        self.num_blocks_per_stage = [2, 3, 5]  # number of dw+pw block pairs per stage
        self.num_stages = len(self.num_blocks_per_stage)
        print(f"Channels: {self.channels}, Blocks per stage: {self.num_blocks_per_stage}")

        self.stem = nn.Sequential(
            Rearrange("bs freq time -> bs 1 freq time"),
            ReparamLargeKernelConv(
                in_chs=1,
                out_chs=self.channels[0],
                kernel_size=15,
                stride=2,
                group_size=0,
                small_kernel=5,
                act_layer=nn.GELU,
            ),
            # pool frequency domain a bit further
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # build conv blocks with minimal branching
        self.conv = nn.Sequential()

        for sdx in range(self.num_stages):
            in_ch = self.channels[sdx]
            out_ch = self.channels[min(sdx + 1, len(self.channels) - 1)]  # Don't go beyond last channel
            num_blocks = self.num_blocks_per_stage[sdx]
            print(in_ch, out_ch, num_blocks)

            for block_idx in range(num_blocks):
                # determine stride and channels for this block
                is_first_block = block_idx == 0
                if is_first_block:
                    # patch embed w/stride (dw+pw)
                    self.conv.add_module(
                        f"patchembed_{sdx}",
                        PatchEmbed(
                            in_chs=in_ch,
                            embed_dim=out_ch,
                            patch_size=5,
                            stride=2,
                        ),
                    )
                    print(f"  PatchEmbed: {in_ch} -> {out_ch}")
                else:
                    # depthwise
                    self.conv.add_module(
                        f"mobileone_dw_{sdx}_{block_idx}",
                        MobileOneBlock(
                            in_chs=in_ch,
                            out_chs=in_ch,
                            kernel_size=3,
                            stride=1,
                            group_size=1,  # implies groups=in_chs
                            num_conv_branches=4,
                        ),
                    )
                    print(f"  MobileOneBlock DW: {in_ch} -> {in_ch}")
                    # pointwise
                    self.conv.add_module(
                        f"mobileone_pw_{sdx}_{block_idx}",
                        MobileOneBlock(
                            in_chs=in_ch,
                            out_chs=in_ch,
                            kernel_size=1,
                            stride=1,
                            group_size=0,  # implies groups=1
                            num_conv_branches=4,
                        ),
                    )
                    print(f"  MobileOneBlock PW: {in_ch} -> {in_ch}")

                # Update in_ch for next iteration
                in_ch = out_ch

        # maxpool freq dimension and prepare for recurrent layer
        self.freq_pool = Reduce("bs ch freq time -> bs time ch", "max")
        # process time dimension w/recurrent layer
        self.recurrent = nn.GRU(self.channels[-1], hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        # avgpool time dimension
        self.time_pool = Reduce("bs time emb -> bs emb", "mean")
        # DropPath (stochastic depth) is good but takes a LONG time to run, double epochs vs Dropout.
        # self.dropout = nn.Dropout(0.3)
        self.dropout = DropPath(0.1)

        # output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.stem(x)  # (bs, ch, freq, time)
        x = self.conv(x)  # (bs, ch, freq, time)
        x = self.freq_pool(x)  # (bs, time, ch)
        out, _ = self.recurrent(x)  # (bs, time, hidden)
        out = self.time_pool(out)  # (bs, hidden)
        out = self.dropout(out)

        return self.fc(out)  # (bs, num_classes)
