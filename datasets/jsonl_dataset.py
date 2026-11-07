# -*- coding: utf-8 -*-
"""OTSL 8 标签 dataset: jsonl 标注 + 同名 png (与 tabeldetr 的数据格式完全一致).

数据形态
    jsonl 每行: {"filename": "PMC...png", "split": "train", "imgid": 123,
                 "html": {"structure": {"tokens": [...]}, "cells": [{"tokens": [...], "bbox": [...]}]}}
    图片: 与 jsonl 同目录的 <filename>

    structure tokens 里没有 otsl 字段, 在 __getitem__ 里现场用
    datasets/convert.py 转成 7 个网格标签 + <eos> (实测 5 万条结构 100% 一致).

为什么是 map-style
    tabeldetr 用的就是这种 (JsonlOffsetIndex 字节偏移 + lazy PIL),
    随机 shuffle / 随机访问 / 按 split 过滤都靠标准 RandomSampler, 不需要内存 buffer.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import get_config                          # noqa: E402
from datasets.convert import structure_to_otsl         # noqa: E402
from datasets.otsl import encode, IGNORE_INDEX  # noqa: E402

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset, DataLoader
    HAS_TORCH = True
except ImportError:                                    # pragma: no cover
    _TorchDataset = object
    HAS_TORCH = False

CFG = get_config()
ZERO = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- jsonl 索引

class JsonlOffsetIndex:
    """jsonl 字节偏移索引: 建索引扫一遍, 读样本时 seek, 不全量进内存."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.offsets = []
        with open(self.path, "rb") as f:
            pos = 0
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)
        self._fh = None
        self._pid = None

    def _handle(self):
        if self._fh is None or self._pid != os.getpid():   # fork 后重开句柄
            self._fh = open(self.path, "rb")
            self._pid = os.getpid()
        return self._fh

    def read(self, i):
        f = self._handle()
        f.seek(self.offsets[i])
        return json.loads(f.readline().decode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __del__(self):
        try:
            if self._fh:
                self._fh.close()
        except Exception:                              # noqa: BLE001
            pass


# ---------------------------------------------------------------- 图像

def load_and_pad_image(path, size=896, fill=255):
    """等比缩放 + 右下白色 pad (表格白底, 比黑边更接近真实分布).

    返回 (PIL RGB, scale, (orig_w, orig_h)).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (0, 0))
    return canvas, scale, (w, h)


# ---------------------------------------------------------------- dataset

class OtslTableDataset(_TorchDataset):
    """jsonl -> (7 网格标签 + <eos> 的 token 序列, 逐 token 的 cell bbox).

    返回 dict:
        img        PIL.Image (img_size x img_size)
        token_ids  list[int]   长度 = rows*(cols+1)
        labels     list[int]   同 token_ids (对齐到槽位由 collate/训练侧做)
        boxes      list[[x0,y0,x1,y1]]  与 token_ids 等长, 无坐标处 (0,0,0,0)
        box_mask   list[int]   1=真实坐标
        rows/cols/filename/imgid/scale/orig_size
    """

    def __init__(self, jsonl_path=None, img_dir=None, cfg=None, split=None,
                 img_size=None, max_len=None, long_policy=None, with_bbox=True):
        cfg = cfg or CFG
        d = cfg.data
        if split is None:
            split = d.get("split")
        jsonl_path = jsonl_path or cfg.paths.train_jsonl
        self.cfg = cfg
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.img_dir = img_dir or cfg.paths.get("img_dir") \
            or os.path.dirname(self.jsonl_path)
        self.img_size = int(img_size if img_size is not None else d.img_size)
        self.max_len = int(max_len if max_len is not None else d.max_len)
        self.long_policy = long_policy or d.long_policy
        self.with_bbox = with_bbox

        self.index = JsonlOffsetIndex(self.jsonl_path)
        if split is None:
            self.ids = list(range(len(self.index)))
        else:
            self.ids = [i for i in range(len(self.index))
                        if self.index.read(i).get("split") == split]
        self.stats = {"seen": 0, "ok": 0, "bad_struct": 0, "too_long": 0, "no_image": 0}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        n = len(self.ids)
        for k in range(n):                        # 坏样本顺延, 避免死循环
            item = self._build(self.index.read(self.ids[(i + k) % n]))
            if item is not None:
                return item
        raise RuntimeError("dataset 里没有可用样本")

    def _build(self, raw):
        self.stats["seen"] += 1
        html = raw.get("html") or {}
        tokens = (html.get("structure") or {}).get("tokens") or []
        cells = html.get("cells") or []
        try:
            toks, R, C, orders, holes = structure_to_otsl(tokens, cells)
        except Exception:
            self.stats["bad_struct"] += 1
            return None
        if holes or not toks or len(toks) != R * (C + 1) + 1:   # +1 = <eos>
            self.stats["bad_struct"] += 1
            return None
        if len(toks) > self.max_len:
            if self.long_policy == "drop":
                self.stats["too_long"] += 1
                return None
            toks = toks[: self.max_len]
            orders = orders[: self.max_len]

        path = os.path.join(self.img_dir, raw["filename"])
        try:
            img, scale, orig_size = load_and_pad_image(path, self.img_size)
        except (OSError, FileNotFoundError):
            self.stats["no_image"] += 1
            return None

        ids = encode(toks)
        if self.with_bbox:
            boxes, mask = [], []
            for order in orders:
                box, m = list(ZERO), 0
                if order >= 0 and order < len(cells):
                    bbox = cells[order].get("bbox")
                    if bbox and len(bbox) >= 4:
                        box = [float(v) * scale for v in bbox[:4]]   # 原图 -> 输入图坐标
                        m = 1
                boxes.append(box)
                mask.append(m)
        else:
            boxes = [list(ZERO) for _ in ids]
            mask = [0] * len(ids)

        self.stats["ok"] += 1
        return {
            "img": img,
            "token_ids": ids,
            "labels": list(ids),
            "boxes": boxes,
            "box_mask": mask,
            "rows": R,
            "cols": C,
            "filename": raw["filename"],
            "imgid": raw.get("imgid", -1),
            "scale": scale,
            "orig_size": orig_size,
        }


# ---------------------------------------------------------------- 装配

def collate_fn(batch):
    """pad 到 batch 内最长. pad 位: label=-100, box=(0,0,0,0), mask=0.

    只产出 labels, 不产出 token_ids: DETR 不吃 token 输入, 而 labels 已经是
    补齐后的 id 序列. 于是 pad 只以 -100 出现, 在 id 空间里不占任何编号.
    """
    if not HAS_TORCH:
        return batch
    L = max(len(b["token_ids"]) for b in batch)
    labels, attn, boxes, bmask = [], [], [], []
    for b in batch:
        n = L - len(b["token_ids"])
        assert len(b["boxes"]) == len(b["token_ids"]) == len(b["box_mask"]), \
            "boxes/box_mask 与 token_ids 长度不一致: %s" % b["filename"]
        labels.append(b["token_ids"] + [IGNORE_INDEX] * n)
        attn.append([1] * len(b["token_ids"]) + [0] * n)
        boxes.append([list(map(float, x)) for x in b["boxes"]] + [[0.0] * 4] * n)
        bmask.append(b["box_mask"] + [0] * n)

    imgs = np.stack([np.asarray(b["img"], dtype=np.float32) / 255.0 for b in batch])
    return {
        "img": torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "box_mask": torch.tensor(bmask, dtype=torch.float32),
        "rows_cols": [(b["rows"], b["cols"]) for b in batch],
        "filename": [b["filename"] for b in batch],
        "imgid": [b.get("imgid", -1) for b in batch],
    }


def build_dataloader(split="train", cfg=None, batch_size=None, num_workers=None,
                     shuffle=None, **dataset_kw):
    """按 config 组装 DataLoader (map-style, shuffle 用标准 RandomSampler)."""
    cfg = cfg or CFG
    d = cfg.data
    if not HAS_TORCH:
        raise RuntimeError("需要 torch 才能建 DataLoader")

    if split == "train":
        jsonl = cfg.paths.train_jsonl
        bs = int(batch_size or d.batch_size)
        shuffle = True if shuffle is None else shuffle
        drop_last = bool(d.get("drop_last", True))
    elif split in ("val", "valid", "eval"):
        jsonl = cfg.paths.val_jsonl
        bs = int(batch_size or d.eval_batch_size)
        shuffle = False if shuffle is None else shuffle
        drop_last = False
    else:
        raise ValueError("split 只能是 train / val, 当前 %r" % split)

    ds = OtslTableDataset(jsonl, cfg=cfg, **dataset_kw)
    nw = int(d.num_workers) if num_workers is None else int(num_workers)
    pm = bool(d.pin_memory) and torch.cuda.is_available()
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last,
                      num_workers=nw, pin_memory=pm, collate_fn=collate_fn,
                      prefetch_factor=(int(d.get("prefetch_factor", 2)) if nw > 0 else None),
                      persistent_workers=nw > 0)


__all__ = ["JsonlOffsetIndex", "load_and_pad_image", "OtslTableDataset",
           "collate_fn", "build_dataloader", "HAS_TORCH"]
