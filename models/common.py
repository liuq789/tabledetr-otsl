# -*- coding: utf-8 -*-
"""DETR 模型用的积木: 双分支输出头 + 二维正弦位置编码.

约定
    token_logits  (B, N, num_classes)   每个槽位的结构标签分类
    boxes         (B, N, 4)            归一化坐标 (0~1), 由 sigmoid 输出
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 输出头

class DualBranchHead(nn.Module):
    """TableMaster 式输出头: token 分类与 box 回归各自一套 norm + 投影.

    两条分支的 Transformer 层在模型里各自堆叠 (shared -> cls 支路 / box 支路),
    这里只负责最后的归一化 + 线性输出, 两边参数不共享.
    box 分支允许一个小 MLP (box_mlp_layers), 默认 1 层就够 —— 深度已经由支路的
    Transformer 层承担.
    """

    def __init__(self, hidden_dim, num_classes, box_param="cxywh", box_mlp_layers=1):
        super().__init__()
        self.norm_cls = nn.LayerNorm(hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)

        self.norm_box = nn.LayerNorm(hidden_dim)
        layers = []
        for _ in range(max(int(box_mlp_layers), 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.LayerNorm(hidden_dim)]
        layers.append(nn.Linear(hidden_dim, 4))
        self.box = nn.Sequential(*layers)

        if box_param == "cxywh":
            # 让 w/h 初始偏小 (sigmoid(-2)≈0.12), 避免一开始就吐出覆盖整图的大框
            with torch.no_grad():
                self.box[-1].bias[2:] = -2.0
        self.box_param = box_param

    def forward(self, x_cls, x_box):
        logits = self.cls(self.norm_cls(x_cls))
        raw = torch.sigmoid(self.box(self.norm_box(x_box)))
        if self.box_param == "xyxy":
            return logits, raw
        cx, cy, w, h = raw.unbind(-1)
        # 训练/推理都不裁剪: 一旦把框夹到 [0,1], 触边后 clamp 的梯度恒为 0,
        # w/h 会被永久锁死在大框. 越界由 L1/GIoU 自己拉回来, 只在画图/导出时裁.
        x0 = cx - w * 0.5
        y0 = cy - h * 0.5
        x1 = cx + w * 0.5
        y1 = cy + h * 0.5
        return logits, torch.stack([x0, y0, x1, y1], dim=-1)


# ---------------------------------------------------------------- 位置编码

def _sine_1d(pos, dim, temperature=10000.0):
    """pos: (N,) -> (N, dim) 的一维正弦位置编码."""
    device = pos.device
    half = dim // 2
    freq = torch.exp(-math.log(temperature) * torch.arange(half, device=device) / max(half, 1))
    ang = pos.float()[:, None] * freq[None, :]
    emb = torch.cat([ang.sin(), ang.cos()], dim=1)
    if emb.shape[1] < dim:                       # dim 为奇数时补零
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], dim - emb.shape[1])], dim=1)
    return emb


def build_2d_sine_pe(h, w, dim):
    """二维正弦位置编码 (h*w, dim): 一半编码行, 一半编码列."""
    dim_h = dim // 2
    dim_w = dim - dim_h
    rows = torch.arange(h).repeat_interleave(w)
    cols = torch.arange(w).repeat(h)
    return torch.cat([_sine_1d(rows, dim_h), _sine_1d(cols, dim_w)], dim=1)


__all__ = ["DualBranchHead", "build_2d_sine_pe"]
