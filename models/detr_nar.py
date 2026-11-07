# -*- coding: utf-8 -*-
"""DETR 式非自回归解码 (ResNet + N 个可学习 query, 严格对齐, 无匈牙利).

结构 (TableMaster 式双分支)
    图像 -> ResNet -> 特征序列 (K/V) ─┬─ cross-attn ─┐
                                      │              │
    N 个可学习 query ──── 共享层 ─────┴──────────────┴─┬─ cls 支路 -> N x 8 (7 网格标签 + eos)
                                                       └─ box 支路 -> N x 4 (cell 框)

特点
    * 共享 2 层后分叉, 分类与框回归各若干层, 各支路都有自己的 cross-attention
    * query self-attention 用因果掩码 (model.query_causal, 默认 true)
    * query i 严格对应第 i 个标签, 逐位交叉熵, 不做二分图匹配
    * 一次前向出整条序列; 超长截断, 不足补 -100 (不算 loss)
    * 层数由 config 控制: model.shared_layers / cls_layers / box_branch_layers
"""
import torch
import torch.nn as nn

from .backbone import build_backbone
from .common import DualBranchHead, build_2d_sine_pe


class DetrNARTSR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        hidden = int(m.hidden_dim)
        self.hidden = hidden
        self.num_slots = int(m.num_queries)
        self.num_classes = int(m.num_classes)
        self.query_causal = bool(m.get("query_causal", True))

        self.backbone = build_backbone(cfg)
        self.query_embed = nn.Embedding(self.num_slots, hidden)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        def make_layer():
            return nn.TransformerDecoderLayer(
                d_model=hidden, nhead=int(m.nheads),
                dim_feedforward=int(m.dim_feedforward),
                dropout=float(m.dropout), activation="gelu",
                batch_first=True, norm_first=True)

        n_shared = int(m.get("shared_layers", 2))
        n_cls = int(m.get("cls_layers", max(int(m.dec_layers) - n_shared, 1)))
        n_box = int(m.get("box_branch_layers", n_cls))
        self.shared_layers = nn.ModuleList([make_layer() for _ in range(n_shared)])
        self.cls_layers = nn.ModuleList([make_layer() for _ in range(n_cls)])
        self.box_layers = nn.ModuleList([make_layer() for _ in range(n_box)])

        self.head = DualBranchHead(hidden, self.num_classes,
                                   box_param=str(m.get("box_param", "cxywh")),
                                   box_mlp_layers=int(m.get("box_layers", 1)))
        self._pe_cache = {}
        self._mask_cache = {}

    def _pe(self, h, w, device):
        key = (h, w, device)
        if key not in self._pe_cache:
            self._pe_cache[key] = build_2d_sine_pe(h, w, self.hidden).to(device)
        return self._pe_cache[key]

    def _tgt_mask(self, n, device, dtype):
        """上三角 -inf 的加性掩码: 位置 i 只能 attend 到 0..i."""
        key = (n, str(device), dtype)
        if key not in self._mask_cache:
            mask = torch.full((n, n), float("-inf"), device=device, dtype=dtype)
            mask = torch.triu(mask, diagonal=1)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, img):
        mem, (h, w) = self.backbone(img)
        mem = mem + self._pe(h, w, mem.device)
        B = img.shape[0]

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        mask = self._tgt_mask(q.shape[1], q.device, q.dtype) if self.query_causal else None

        x = q
        for layer in self.shared_layers:
            x = layer(x, mem, tgt_mask=mask)

        xc = x
        for layer in self.cls_layers:
            xc = layer(xc, mem, tgt_mask=mask)
        xb = x
        for layer in self.box_layers:
            xb = layer(xb, mem, tgt_mask=mask)

        logits, boxes = self.head(xc, xb)
        return {"token_logits": logits, "boxes": boxes, "aux": []}
