# -*- coding: utf-8 -*-
"""DETR 式表格结构识别: ResNet backbone + 可学习 query 并行解码.

    图像 -> ResNet -> 特征序列 (K/V)
    N 个可学习 query ─ 共享层 -> cls 支路 / box 支路 (各带 cross-attention)
                     -> N x 8 (7 网格标签 + eos) / N x 4 (cell 框)

结构取自 tabeldetr/models/detr_nar.py, 未做改动;
唯一区别是 num_classes = 8 (OTSL 7 个网格标签 + <eos>), 由 config.yaml 的 model.num_classes 控制.
"""
from .detr_nar import DetrNARTSR


def build_model(cfg):
    return DetrNARTSR(cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


__all__ = ["DetrNARTSR", "build_model", "count_params"]
