# Learned Queries for Efficient Local Attention
# Modified from the JAX Implementation of QnA (20220830)
# https://github.com/moabarar/qna/blob/main/layers/qna.py
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# Not used in the journal paper, only for reference.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops
import math

class FusedKQnA(nn.Module):
    
    def __init__(self, n_q, n_channels, n_heads, ksize, stride, padding, qna_activation):
        
        super().__init__()
        self.n_q = n_q
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.head_channels = n_channels // n_heads
        self.qna_activation = qna_activation

        
        self.proj_k = nn.Linear(self.n_channels, self.n_channels * stride, bias=False)
        self.proj_v = nn.Linear(self.n_channels, self.n_channels * stride, bias=False)
        self.proj_out = nn.Conv2d(self.n_channels * stride, self.n_channels * stride, 1, 1, 0, bias=False)
        self.scale = self.head_channels ** -0.5
        self.q_param = nn.Parameter(torch.empty(self.n_q, self.n_channels * stride))
        trunc_normal_(self.q_param, std=math.sqrt(1.0 / self.head_channels))

        self.attn_scale = nn.Parameter(torch.empty(1, 1, self.ksize ** 2, self.n_q * self.n_heads * stride))
        nn.init.normal_(self.attn_scale, std=0.02)
        
        self.rpb_table = nn.Parameter(torch.empty(self.ksize ** 2, self.n_q * self.n_heads * stride))
        trunc_normal_(self.rpb_table, std=0.02)
        
    def forward(self, x):
        # assert not torch.isnan(x).any(), 'NaN in x'
        B, C, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        N = H * W
        q = self.q_param[None, ...].expand(B, self.n_q, C * self.stride)
        # assert not torch.isnan(q).any(), 'NaN in q'
        q = einops.rearrange(q, 'b q (h c) -> b h q c', h=self.n_heads * self.stride, c=self.head_channels, q=self.n_q)
        # q_norm_l2 = torch.linalg.vector_norm(q, ord=2, dim=3, keepdim=True)
        # q = q / (q_norm_l2 + 1e-6)
        k = self.proj_k(x)
        B, N, C = k.size()
        # assert not torch.isnan(k).any(), 'NaN in k'
        k = einops.rearrange(k, 'b k (h c) -> b h k c', h=self.n_heads * self.stride, c=self.head_channels, k=N)
        q = q * self.scale
        qkT = torch.einsum('b h q c, b h k c -> b h q k', q, k)
        qkT = einops.rearrange(qkT, 'b h q k -> b k q h')
        
        # assert not torch.isnan(qkT).any(), 'NaN in qkT'
        
        v = self.proj_v(x)
        attn_scale = self.attn_scale.reshape(self.ksize, self.ksize, 1, self.n_q * self.n_heads * self.stride)
        rpb = self.rpb_table.reshape(self.ksize, self.ksize, 1, self.n_q * self.n_heads * self.stride)
        
        if self.qna_activation == 'exp':
            cost_exp = torch.exp(qkT - qkT.max().detach())
        elif self.qna_activation == 'sigmoid':
            cost_exp = qkT.sigmoid()
        elif self.qna_activation == 'linear':
            cost_exp = qkT
        
        # assert not torch.isnan(cost_exp).any(), 'NaN in cost_exp'

        v_cost_exp = cost_exp[..., None] * v.reshape(B, N, 1, self.n_heads * self.stride, self.head_channels)
        # v_cost_exp : B N n_q h ch
        v_cost_exp = v_cost_exp.reshape(B, N, self.n_q * self.n_heads * self.stride * self.head_channels)
        
        if self.qna_activation == 'exp':
            rpb_exp = torch.exp(rpb - rpb.max().detach())
        elif self.qna_activation == 'sigmoid':
            rpb_exp = rpb.sigmoid()
        elif self.qna_activation == 'linear':
            rpb_exp = rpb
        
        summation_kernel = (rpb_exp * attn_scale).repeat_interleave(self.head_channels, dim=3)
        # summation_kernel : [kh, kw, 1, n_q * h * Ch] HWIO
        v_cost_exp_ = einops.rearrange(v_cost_exp, 'b (h w) c -> b c h w', h=H, w=W)
        v_kern_ = einops.rearrange(summation_kernel, 'h w i o -> o i h w')
        sum_num = F.conv2d(
                v_cost_exp_,
                v_kern_,
                stride=self.stride,
                padding=self.padding,
                groups=self.n_q * self.n_channels * self.stride
            ) # B Nq*h*Ch H 
        sum_num = einops.rearrange(
            sum_num,
            'b (q g c) h w -> b q g c h w',
            q=self.n_q, g=self.n_heads * self.stride, c=self.head_channels)
        cost_exp_ = einops.rearrange(
            cost_exp,
            'b (h w) q c -> b (q c) h w',
            h=H,
            w=W,
            q=self.n_q,
            c=self.n_heads * self.stride) # B Nq*h H W
        kern_ = einops.rearrange(rpb_exp, 'h w i o -> o i h w')
        
        sum_den = F.conv2d(
                cost_exp_,
                kern_,
                stride=self.stride,
                padding=self.padding,
                groups=self.n_q * self.n_heads * self.stride
            ) # B Nq*h H W
        sum_den = einops.rearrange(
            sum_den,
            'b (q g c) h w -> b q g c h w',
            q=self.n_q, g=self.n_heads * self.stride, c=1)
        H = H // self.stride
        W = W // self.stride
        out = (sum_num / sum_den).sum(dim=1).reshape(B, C, H, W)
        out = self.proj_out(out)
        return out, None, None
    
