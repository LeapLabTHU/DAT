#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# Originated from https://github.com/SHI-Labs/NATTEN/blob/main/src/natten/natten2d.py
# --------------------------------------------------------

import warnings
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from natten.functional import is_fna_enabled, na2d, na2d_av, na2d_qk, enable_fna, enable_autotuner
from natten.utils import check_all_args

# enable_fna()
# enable_autotuner()
# assert is_fna_enabled(), "Fused neighborhood attention is not enabled."

class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int | Tuple[int, int],
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dilation: int | Tuple[int, int] = 1,
        is_causal: bool | Tuple[bool, bool] = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
    ):
        super().__init__()
        kernel_size, dilation, is_causal = check_all_args(
            2, kernel_size, dilation, is_causal
        )
        if any(is_causal) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads, (2 * kernel_size[0] - 1), (2 * kernel_size[1] - 1)
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                warnings.warn(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet."
                )

            qkv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 1, 2, 4, 5)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, H, W, C)

        else:
            qkv = (
                self.qkv(x)
                .reshape(B, H, W, 3, self.num_heads, self.head_dim)
                .permute(3, 0, 4, 1, 2, 5)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj_drop(self.proj(x))
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, None, None

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )