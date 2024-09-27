# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------
INTERPOLATE_MODE = 'bilinear'

import numpy as np
import math

import torch

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange
from functools import partial
from torch import nn, einsum
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.cnn import build_norm_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

__all__ = [
    "mpvit_tiny",
    "mpvit_xsmall",
    "mpvit_small",
    "mpvit_base",
]


def sep_prompt(x, prompts_len):
    spa_prompts = x[:, :prompts_len, :]
    x = x[:, prompts_len:, :]
    return spa_prompts, x


BatchNorm2d = nn.SyncBatchNorm


def _cfg_mpvit(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a. MLP) class."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            act_layer=None,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.conv = torch.nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
            self,
            in_chans=3,
            embed_dim=768,
            patch_size=16,
            stride=1,
            pad=0,
            act_layer=nn.Hardswish,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # TODO : confirm whether act_layer is effective or not
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    def __init__(self, embed_dim, num_path=4, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg=norm_cfg,
                )
                for idx in range(num_path)
            ]
        )

        # scale

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class ConvRelPosEncs(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        # b,h不动 k的n,k→k，n @ v的 n,v 相当于k.transpose(-2, -1)@ v 整个就是q@k_softmax.transpose(-2, -1)@v
        # 原(q @ k.transpose(-2, -1))@v
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        #  q@k_softmax_T_dot_v(-2,-1)
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class taskFactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size, spa_prompts):
        B, N, C = x.shape
        nW = B // spa_prompts.shape[0]
        prompts_len = 2
        short_x = x
        # B, N, C = short_x.shape
        # qkv = (
        #     self.qkv(short_x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        #     .contiguous()
        # )  # Shape: [3, B, h, N, Ch].
        # q, k, _ = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].
        # # crpe = self.crpe(q, v, size=size).permute(0, 2, 1, 3)# Shape: [B, h, N, Ch]→[B,N,h,ch].
        # # crpe = torch.cat([spa_prompts, crpe], dim=1).permute(0, 2, 1, 3) # Shape: [B,N,h,ch]→[B,h, N,ch].其中N=P+H*W
        # crpe = self.crpe(q, k, size=size)  # Shape: [B, h, N, Ch]
        # crpe = (crpe @ crpe.transpose(-2, -1))  # [B,h, N, N]
        x = torch.cat([spa_prompts, x], dim=1)
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))  # Shape: [3, B, h, N, Ch].
        q, k, v0 = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].
        # softmax?

        factor_weight = (q @ k.transpose(-2, -1))  # Shape: [B, h, N, N]
        factor_att = factor_weight * self.scale

        # Convolutional relative position encoding.


        factor_att[:, :, prompts_len:, prompts_len:] = factor_att[:, :, prompts_len:, prompts_len:]
        attn = self.softmax(factor_att)
        attn = self.attn_drop(attn)

        x = (attn @ v0).transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        spa_prompts, x = sep_prompt(x, prompts_len)
       # spa_prompts = spa_prompts.reshape(B // nW, nW, prompts_len, C).mean(dim=1)


        return x, factor_weight, spa_prompts


class MHCABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        # x.shape = [B, N, C]

        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class taskBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
            chan_embed_dim=450,
            pixel_no=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = taskFactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.token_trans = nn.Linear(dim, chan_embed_dim)
        self.chan_q = nn.Linear(chan_embed_dim, chan_embed_dim, bias=qkv_bias)
        self.chan_kv = nn.Linear(pixel_no, chan_embed_dim * 2, bias=qkv_bias)
        self.chan_scale = chan_embed_dim ** -0.5

    def forward(self, x, size, task_prompts):
        # x.shape = [B, N, C]
        prompts_len = 2
        H, W = size
        B, N, C = x.shape
        assert N == H * W, "input feature has wrong size"

        if self.cpe is not None:
            x = self.cpe(x, size)
        ori_task_prompts = task_prompts
        spa_prompts = self.norm1(task_prompts)
        chan_prompts = self.token_trans(task_prompts)
        shortcut = x
        cur = self.norm1(x)

        attn_cur, factor_weight, spa_prompts = self.factoratt_crpe(cur, size, spa_prompts)

        task_prompts = spa_prompts
        factor_weight = factor_weight[:, :, :prompts_len, prompts_len:]  # (B*nW, nheads, T, wh*ww)
        factor_weight = factor_weight.view(B, -1, prompts_len,H,W)
        raw_spa_attn = factor_weight

        chan_x = attn_cur
        # print("chan_x", chan_x.shape)
        chan_x = chan_x.permute(0,2,1) # (B, C, HxW)
        q = self.chan_q(chan_prompts)
        kv = self.chan_kv(chan_x).reshape(B, C, 2, -1)  # (B, C, 2, chant_dim)
        k, v = kv[:, :, 0, :], kv[:, :, 1, :]  # (B, C, chant_dim)

        raw_chan_attn = (q @ k.transpose(-2, -1))  # (B, T, C)
        # print("raw_chan_attn", raw_chan_attn.shape)
        attn = raw_chan_attn * self.chan_scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        chan_x = (attn @ v)  # (B, T, chant_dim)

        raw_attn = [raw_spa_attn, raw_chan_attn]


        x = shortcut + self.drop_path(attn_cur)
        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))

        task_prompts = ori_task_prompts + self.drop_path(task_prompts)
        task_prompts = task_prompts + self.drop_path(self.mlp(self.norm2(task_prompts)))

        return x, factor_weight, task_prompts


class MHCAEncoder(nn.Module):
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class taskEncoder(nn.Module):
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={3: 2, 5: 3, 7: 3},
            pixel_no=None,
    ):
        super().__init__()
        pixel_no = pixel_no
        self.num_layers = num_layers
       # self.cpe = ConvPosEnc(dim, k=3)
        self.cpe = ConvPosEnc(dim , k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.task_layers = nn.ModuleList(
            [
                taskBlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                    chan_embed_dim=450,
                    pixel_no=pixel_no,
                )
                for idx in range(1)
            ]
        )

    def forward(self, x, size, task_prompts):
        H, W = size
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.task_layers:
            x, factor_weight, task_prompts = layer(x, (H, W), task_prompts)
            # x = layer(x, (H, W),task_prompts)

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x, factor_weight, task_prompts


class ResBlock(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = build_norm_layer(norm_cfg, hidden_features)[1]
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            norm_cfg=dict(type="BN"),
            drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        self.aggregate = Conv2d_BN(
            embed_dim * (num_path + 1),
            out_embed_dim,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, inputs):
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out, att_outputs


class task_stage(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            drop_rate=0,
            proj_drop=0.0,
            norm_cfg=dict(type="BN"),
            drop_path_list=[],
            pixel_no=None,
    ):
        super().__init__()
        final_embed_dim = 450
        prompts_len = 2
        self.task_prompts = nn.Parameter(torch.ones(prompts_len, out_embed_dim))
        trunc_normal_(self.task_prompts, mean=1., std=1.)
        self.proj_drop = nn.Dropout(proj_drop)
        self.TASKSNAMES = ['depth', 'pose']
        self.multi_scale_fuse = nn.ModuleDict(
            {t: nn.Conv2d(out_embed_dim, out_embed_dim, kernel_size=3, padding=1) for t in self.TASKSNAMES if
             t != '3ddet'})
        self.fea_fuse = nn.ModuleList()
        self.fea_decode_spa = nn.ModuleList()
        self.fea_decode_chan = nn.ModuleList()

        self.norm = nn.LayerNorm(out_embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.fea_fuse.append(nn.ModuleDict())
        self.fea_decode_spa.append(nn.ModuleDict())
        self.fea_decode_chan.append(nn.ModuleDict())
        for task in self.TASKSNAMES:
            self.fea_fuse[0][task] = nn.Sequential(nn.Conv2d(out_embed_dim*2, out_embed_dim, kernel_size=1),nn.Conv2d(out_embed_dim, out_embed_dim, kernel_size=3, padding=1))
            self.fea_fuse[0][task+"BatchNorm2d"] = BatchNorm2d(out_embed_dim)
            self.fea_fuse[0][task+"Conv2d2"] = nn.Sequential(nn.GELU(),nn.Conv2d(out_embed_dim, out_embed_dim, kernel_size=3, padding=1))

            self.fea_decode_spa[0][task] = nn.Sequential(
                    nn.Conv2d(out_embed_dim, out_embed_dim, kernel_size=1, padding=0))
            self.fea_decode_chan[0][task] = nn.Sequential(
                    nn.Conv2d(out_embed_dim, out_embed_dim, kernel_size=1, padding=0))

        self.task_blks = nn.ModuleList(
            [taskEncoder(
                out_embed_dim,
                num_layers,
                num_heads,
                mlp_ratio,
                drop_path_list=drop_path_list,
                pixel_no=pixel_no,

            )
                # 一个block1个多任务处理
                for _ in range(1)
            ]
        )
    def forward(self, inputs):


        outs = inputs.flatten(2).transpose(1, 2)  # BCHW→BNC
        outs = self.norm(outs)
        outs = self.pos_drop(outs)  # BNC
        task_prompts = self.task_prompts[None].expand(outs.shape[0], -1, -1)
        # self.all_tasks 需要p.TASKS.NAMES
        task_fea = {task: [] for task in self.TASKSNAMES}
        info = {}  # pass information through the pipeline
        _, _, H, W = inputs.shape
        size = (H, W)

        attn_outs, factor_weight, task_prompts = self.task_blks[0](outs, size, task_prompts)

        cur_task_fea = self.cal_task_feature(attn_outs, factor_weight, size)
        for t_idx, task in enumerate(self.TASKSNAMES):
            task_fea[task].append(cur_task_fea[task])
        new_task_fea = {}
        for task in self.TASKSNAMES:
            target_scale = task_fea[task][0].shape[-2:]
            _task_fea = sum([F.interpolate(_, target_scale, mode=INTERPOLATE_MODE) for _ in task_fea[task]])
            _task_fea = self.multi_scale_fuse[task](_task_fea)
            new_task_fea[task] = _task_fea

        return new_task_fea

    def cal_task_feature(self, x, factor_weight, size):
        ''' Calculate task feature at each stage
            '''

        H, W = size
        #x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        task_fea = {}
        chan_task_fea = {}
        spa_attn = factor_weight
        prompt_len = 1
        TASKS_NAMES = {"depth", "pose"}

        for t_idx, task in enumerate(TASKS_NAMES):
            # task feature extraction with spatial attention
            cur_attn_weight = spa_attn[:, :, t_idx * prompt_len:(t_idx + 1) * prompt_len,
                              :,:]  # (b, nheads, prompt_len, h*w)
           # cur_attn_weight = rearrange(cur_attn_weight, 'b nh np h w -> b (nh np) h w', h=H,
                                       # w=W)

            bs, nheads = cur_attn_weight.shape[0:2]
            cur_task_fea = []
            # 没找到self.p.backbone_channels
            head_channel_no = x.shape[1]// nheads
            for hea in range(nheads):
                cur_head_attn = cur_attn_weight[:, hea:hea + 1, :, :,:].squeeze(dim=2)
                cur_task_fea.append(cur_head_attn * x[:, head_channel_no * hea:head_channel_no * (hea + 1), :, :])
            cur_task_fea = torch.cat(cur_task_fea, dim=1) + x
            # if task != '3ddet':
            #     cur_task_fea = F.interpolate(cur_task_fea, scale_factor=2, mode=INTERPOLATE_MODE, align_corners=False)
            cur_task_fea = self.fea_decode_spa[0][task](cur_task_fea)
            task_fea[task] = cur_task_fea
            cur_task_fea = cur_task_fea + x
            cur_task_fea = self.fea_decode_chan[0][task](cur_task_fea)
            chan_task_fea[task] = cur_task_fea
            task_fea[task] = cur_task_fea
            combined_fea = torch.cat([task_fea[task], chan_task_fea[task]], dim=1)
            combined_fea = self.fea_fuse[0][task](combined_fea)
           # combined_fea =self.fea_fuse[0][task + "BatchNorm2d"](combined_fea)
            combined_fea =self.fea_fuse[0][task + "Conv2d2"](combined_fea)
            task_fea[task] = combined_fea

        return task_fea


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur: cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


@BACKBONES.register_module()
class MPViT(nn.Module):
    """Multi-Path ViT class."""

    def __init__(
            self,
            num_classes=80,
            in_chans=3,
            num_stages=4,
            num_layers=[1, 1, 1, 1],
            mlp_ratios=[8, 8, 4, 4],
            num_path=[4, 4, 4, 4],
            embed_dims=[64, 128, 256, 512],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.2,
            norm_cfg=dict(type="BN"),
            norm_eval=False,
            pixel_no = [7680, 1920, 480, 120],
            pretrained=None,
    ):
        super().__init__()
        prompts_len = 2
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.conv_norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self. pixel_no= pixel_no

        # self.task_prompts = nn.Parameter(torch.ones(prompts_len, embed_dims[0]))

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList(
            [
                Patch_Embed_stage(
                    embed_dims[idx],
                    num_path=num_path[idx],
                    isPool=True,
                    norm_cfg=self.conv_norm_cfg,
                )
                for idx in range(self.num_stages)
            ]
        )

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList(
            [
                MHCA_stage(
                    embed_dims[idx],
                    embed_dims[idx + 1]
                    if not (idx + 1) == self.num_stages
                    else embed_dims[idx],
                    num_layers[idx],
                    num_heads[idx],
                    mlp_ratios[idx],
                    num_path[idx],
                    norm_cfg=self.conv_norm_cfg,
                    drop_path_list=dpr[idx],
                )
                for idx in range(self.num_stages)
            ]
        )
        self.task_stages = nn.ModuleList(
            [
                task_stage(
                    embed_dims[idx],
                    embed_dims[idx + 1]
                    if not (idx + 1) == self.num_stages
                    else embed_dims[idx],
                    num_layers[idx],
                    num_heads[idx],
                    mlp_ratios[idx],
                    num_path[idx],
                    norm_cfg=self.conv_norm_cfg,
                    drop_path_list=dpr[idx],
                    pixel_no=pixel_no[idx],
                )
                for idx in range(self.num_stages)
            ]
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):

        # x's shape : [B, C, H, W]
        outs = []
        depth_fea = []
        pose_fea = []
        # 第一步
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]
        outs.append(x)

       # task_prompts = self.task_prompts[None].expand(outs.shape[0], -1, -1)
        for idx in range(self.num_stages):
            # 编码
            att_inputs = self.patch_embed_stages[idx](x)
            # outs.append(att_inputs)
            x, ff = self.mhca_stages[idx](att_inputs)
            outs.append(x)
            if idx > 1:
                new_task_fea = self.task_stages[idx](x)
                depth_fea.append(new_task_fea['depth'])
                if idx==self.num_stages-1:
                    pose_fea.append(new_task_fea['pose'])
        return outs,depth_fea,pose_fea

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MPViT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def mpvit_tiny(**kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    checkpoint = torch.load('./ckpt/mpvit_xsmall.pth', map_location=lambda storage, loc: storage)['model']
    logger = get_root_logger()
    load_state_dict(model, checkpoint, strict=False, logger=logger)
    del checkpoint
    del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_small(**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    # checkpoint = torch.load('./ckpt/mpvit_small.pth', map_location=lambda storage, loc: storage)['model']
    # logger = get_root_logger()
    # load_state_dict(model, checkpoint, strict=False, logger=logger)
    # del checkpoint
    # del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    - MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model
