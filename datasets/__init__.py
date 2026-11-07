# -*- coding: utf-8 -*-
"""OTSL 数据集包: 7 个网格标签 + <eos> (无 thead/tbody).

    from datasets import OTSL_TOKENS, parse_otsl, build_grid, grid_to_html
    from datasets import OtslTableDataset, collate_fn, build_dataloader
"""
from .otsl import (FCEL, ECEL, SPAN, LCEL, UCEL, XCEL, NL, EOS, PAD,
                   OTSL_TOKENS, STOI, ITOS, NUM_CLASSES, IGNORE_INDEX,
                   CELL_TOKENS, OCCUPY_TOKENS, normalize_token, parse_otsl,
                   encode, decode, Cell, Grid, infer_shape, build_grid,
                   grid_to_html, grid_to_tokens, grid_to_six_tokens,
                   to_span_scheme, to_fixed_matrix, CFG)
from .convert import structure_to_grid, grid_to_otsl_tokens, structure_to_otsl
from .jsonl_dataset import (JsonlOffsetIndex, OtslTableDataset, load_and_pad_image,
                            collate_fn, build_dataloader)

__all__ = [
    "FCEL", "ECEL", "SPAN", "LCEL", "UCEL", "XCEL", "NL", "EOS", "PAD",
    "OTSL_TOKENS", "STOI", "ITOS", "NUM_CLASSES", "IGNORE_INDEX",
    "CELL_TOKENS", "OCCUPY_TOKENS", "normalize_token", "parse_otsl",
    "encode", "decode", "Cell", "Grid", "infer_shape", "build_grid",
    "grid_to_html", "grid_to_tokens", "grid_to_six_tokens",
    "to_span_scheme", "to_fixed_matrix", "CFG",
    "structure_to_grid", "grid_to_otsl_tokens", "structure_to_otsl",
    "JsonlOffsetIndex", "OtslTableDataset", "load_and_pad_image",
    "collate_fn", "build_dataloader",
]
