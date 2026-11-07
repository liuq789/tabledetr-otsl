# -*- coding: utf-8 -*-
"""推理: 表格图片 -> (7 网格标签 + <eos> 的 tokens, 逐 cell bbox) -> JSON / HTML / 标注图.

模型输出是 500 个槽位上的 (8 类标签, 4 维框), 一次前向并行给出整条 OTSL 序列。
本脚本负责把这份输出还原成可用的结构化结果:

    图片 -> load_and_pad_image -> 模型 -> argmax + 遇到 <eos> 截断
        -> build_grid 重建网格 -> 每个 cell 的 (r, c, rowspan, colspan)
        -> bbox 从 0~1 反算回原图像素
        -> JSON (+ HTML / 画框图)

用法
    python infer.py --image ../tabeldetr/data/val/PMC1234_001.png
    python infer.py --dir ../tabeldetr/data/val --limit 20
    python infer.py --image x.png --ckpt outputs/detr_otsl/best.pt --no-viz

默认读 outputs/detr_otsl/best.pt, 结果写到 outputs/infer/。
结构里不含文字 (7 个标签只描述网格), 需要文字的话按 bbox 裁图再走 OCR。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import _wrap, get_config, parse_overrides, summary      # noqa: E402
from datasets import (CELL_TOKENS, ECEL, EOS, FCEL, ITOS, SPAN,     # noqa: E402
                      STOI, build_grid, grid_to_html, load_and_pad_image)
from models import build_model, count_params                        # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# 画框配色 (RGB): 有内容 / 空 / 合并区左上角
COLORS = {FCEL: (37, 99, 235), ECEL: (148, 163, 184), SPAN: (220, 104, 3)}


# ---------------------------------------------------------------- 模型

def load_model(ckpt_path, cfg, device):
    """从 checkpoint 重建模型.

    优先用 checkpoint 里存的 cfg: 训练时的 hidden/层数/num_classes 都在里面,
    用当前 config.yaml 重建容易因为后来改过参数而 shape mismatch.
    """
    if not os.path.exists(ckpt_path):
        raise SystemExit("checkpoint 不存在: %s" % ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("cfg"):
        cfg = _wrap(ck["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])              # 严格: 键名/形状不符会直接抛错
    model.to(device).eval()
    total, _ = count_params(model)
    print("ckpt    : %s (epoch %s, best token_acc %s)" % (
        ckpt_path, ck.get("epoch", "?"),
        ("%.4f" % ck["best"]) if ck.get("best") is not None else "?"))
    print("model   : %.2fM 参数 | classes %d | queries %d | img %d" % (
        total / 1e6, cfg.model.num_classes, cfg.model.num_queries, cfg.data.img_size))
    return model, cfg


@torch.no_grad()
def predict(model, img_tensor, device, amp=False):
    """返回 (token_ids [N], token_probs [N], boxes [N,4] 归一化 0~1)."""
    x = img_tensor.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
        out = model(x)
    logits = out["token_logits"][0].float()          # (N, 8)
    probs = logits.softmax(-1)
    score, ids = probs.max(-1)
    boxes = out["boxes"][0].float()                  # 训练侧不裁剪, 这里也不裁
    return ids.cpu(), score.cpu(), boxes.cpu()


def cut_at_eos(ids, probs_tensor):
    """遇到第一个 <eos> 截断; 没预测到 eos 就保留全部槽位并返回 False."""
    eos_id = STOI[EOS]
    hit = (ids == eos_id).nonzero(as_tuple=False)
    if len(hit):
        end = int(hit[0])
        return ids[:end], probs_tensor[:end], True
    return ids, probs_tensor, False


# ---------------------------------------------------------------- 后处理

def boxes_to_pixels(boxes, img_size, scale, size):
    """0~1 的框 -> 原图像素; 训练时 img 是等比缩放后放在 img_size 画布左上角."""
    w, h = size
    out = []
    for b in boxes:
        x0, y0, x1, y1 = [float(v) * img_size / max(scale, 1e-6) for v in b]
        x0, x1 = sorted((max(0.0, min(x0, w)), max(0.0, min(x1, w))))
        y0, y1 = sorted((max(0.0, min(y0, h)), max(0.0, min(y1, h))))
        out.append([round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)])
    return out


def tokens_to_cells(tokens, boxes_px, scores):
    """标签序列 -> 网格 + 每个 cell 的框.

    build_grid 给出的 grid.cells 就是按"开启单元格的顺序"排的, 而同一顺序正好
    对应序列里的 <fcel>/<ecel>/<span>, 所以直接按顺序配对 (第 i 个 cell 标签 <-> 第 i 个框).
    """
    cell_idx = [i for i, t in enumerate(tokens) if t in CELL_TOKENS]
    try:
        grid = build_grid(tokens, strict=False)
    except Exception as exc:                                  # noqa: BLE001
        return None, [], "build_grid 失败: %s" % exc

    cells = []
    for k, cell in enumerate(grid.cells):
        ti = cell_idx[k] if k < len(cell_idx) else None
        cells.append({
            "id": k,
            "r": cell.r, "c": cell.c,
            "rowspan": cell.rowspan, "colspan": cell.colspan,
            # 合并区可能由 <span> 直接标出, 也可能靠后面的 lcel/ucel 撑开, 两种都算 span
            "kind": ("span" if (cell.is_span or cell.area > 1)
                     else ("fcel" if cell.has_content else "ecel")),
            "token_index": ti,
            "bbox": boxes_px[ti] if ti is not None and ti < len(boxes_px) else None,
            "score": round(float(scores[ti]), 4) if ti is not None and ti < len(scores) else None,
        })
    warn = None
    if len(cell_idx) != len(grid.cells):
        warn = "cell 标签 %d 个, 但网格里只有 %d 个 cell (序列可能有残缺)" % (
            len(cell_idx), len(grid.cells))
    return grid, cells, warn


def draw_boxes(src_path, cells, out_path):
    """把每个 cell 的框画回原图, 编号与 JSON 里的 id 一致."""
    img = Image.open(src_path).convert("RGB")
    canvas = Image.new("RGB", img.size, (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for c in cells:
        b = c.get("bbox")
        if not b:
            continue
        color = COLORS.get(c["kind"], (37, 99, 235))
        d.rectangle(b, outline=color, width=2)
        d.text((b[0] + 2, max(0.0, b[1] - 11)), str(c["id"]), fill=color)
    canvas.save(out_path)
    return out_path


# ---------------------------------------------------------------- 单图 / 批量

def infer_one(model, cfg, device, img_path, out_dir, amp=False,
              save_viz=True, save_html=False):
    img_size = int(cfg.data.img_size)
    pil, scale, (ow, oh) = load_and_pad_image(img_path, img_size)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]

    t0 = time.time()
    ids, scores, boxes = predict(model, x, device, amp)
    ids, scores, hit_eos = cut_at_eos(ids, scores)

    tokens = [ITOS[int(i)] for i in ids]
    boxes_px = boxes_to_pixels(boxes[:len(tokens)], img_size, scale, (ow, oh))
    grid, cells, warn = tokens_to_cells(tokens, boxes_px, scores)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    name = os.path.basename(img_path)
    if grid is not None:
        html = grid_to_html(grid, wrap=True)
    else:
        html = None
    if not hit_eos:
        warn = (warn + " | " if warn else "") + \
            "序列未出现 <eos>, 已用满 %d 个槽位 (可能是超长表或被截断)" % len(tokens)

    result = {
        "filename": name,
        "image_size": [ow, oh],
        "input_size": img_size,
        "scale": round(float(scale), 4),
        "ckpt": None,
        "rows": grid.rows if grid else None,
        "cols": grid.cols if grid else None,
        "n_cells": len(cells),
        "n_tokens": len(tokens),
        "hit_eos": bool(hit_eos),
        "mean_score": round(float(np.mean([s for s in scores.tolist()] or [0.0])), 4),
        "tokens": [t[1:-1] for t in tokens],          # 去掉尖括号, 便于人读
        "html": html,
        "cells": cells,
        "warning": warn,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    extras = []
    if save_viz:
        extras.append(draw_boxes(img_path, cells, os.path.join(out_dir, stem + "_boxes.png")))
    if save_html and html:
        hp = os.path.join(out_dir, stem + ".html")
        with open(hp, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><meta charset='utf-8'>\n"
                    "<style>body{font-family:Segoe UI,Microsoft YaHei;padding:16px}"
                    "td{border:1px solid #98a2b3;min-width:28px;height:20px;padding:2px 6px}"
                    "table{border-collapse:collapse}</style>\n"
                    "<h3>%s · %sx%s</h3>\n%s\n"
                    % (name, result["rows"], result["cols"], html))
        extras.append(hp)

    dt = time.time() - t0
    print("%-42s %sx%s | cells %-4d | score %.3f | %s | %.0f ms"
          % (name, result["rows"], result["cols"], len(cells),
             result["mean_score"], "eos" if hit_eos else "无eos", dt * 1000))
    if warn:
        print("  ! %s" % warn)
    result["_paths"] = {"json": json_path, "extras": extras}
    return result


def warn_cfg_drift(cli_cfg, ck_cfg):
    """推理必须用训练时的结构参数, 但 --set 覆盖会悄悄失效, 这里显式提示."""
    keys = [("data", "img_size"), ("model", "hidden_dim"), ("model", "num_classes"),
            ("model", "num_queries"), ("model", "backbone"), ("model", "out_stride"),
            ("model", "shared_layers"), ("model", "cls_layers"), ("model", "box_branch_layers")]
    diff = []
    for sec, k in keys:
        a = (cli_cfg.get(sec) or {}).get(k)
        b = (ck_cfg.get(sec) or {}).get(k)
        if a is not None and b is not None and a != b:
            diff.append("%s.%s: %s -> %s" % (sec, k, a, b))
    if diff:
        print("! 结构参数以 ckpt 为准, 下面这些本次传入的值被忽略:")
        for d in diff:
            print("    %s" % d)


def collect_images(args):
    if args.image:
        return [args.image]
    if args.dir:
        files = sorted(p for p in glob.glob(os.path.join(args.dir, "*"))
                       if os.path.splitext(p)[1].lower() in IMG_EXTS)
        return files[:args.limit] if args.limit else files
    raise SystemExit("需要 --image 或 --dir (或用 --list 从 jsonl 取)")


def main():
    ap = argparse.ArgumentParser(description="表格结构推理 (OTSL 7 网格标签 + eos)")
    ap.add_argument("-c", "--config", default=None, help="config.yaml 路径")
    ap.add_argument("--set", dest="overrides", nargs="*", default=None,
                    help="覆盖配置, 如 data.img_size=896")
    ap.add_argument("--ckpt", default=None,
                    help="默认 <outputs>/detr_otsl/best.pt")
    ap.add_argument("--image", default=None, help="单张图片")
    ap.add_argument("--dir", default=None, help="图片目录 (配合 --limit)")
    ap.add_argument("--list", default=None,
                    help="从 jsonl 读 filename 列表 (如 val 集), 与 --dir 合用")
    ap.add_argument("--limit", type=int, default=8, help="--dir 时最多处理多少张")
    ap.add_argument("--out", default=None, help="输出目录, 默认 outputs/infer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="混合精度推理")
    ap.add_argument("--no-viz", action="store_true", help="不画框图")
    ap.add_argument("--html", action="store_true", help="额外存一份 HTML 表格")
    args = ap.parse_args()

    cfg = get_config(args.config, parse_overrides(args.overrides), reload=True)
    out_dir = args.out or os.path.join(cfg.project.output_dir, "infer")
    ckpt = args.ckpt or os.path.join(cfg.project.output_dir, "detr_otsl", "best.pt")
    device = torch.device(args.device)

    if args.list:                                   # 用 jsonl 的 filename 当图片清单
        img_dir = args.dir
        if not img_dir:                             # 没给 --dir 就用 jsonl 所在目录
            img_dir = os.path.dirname(os.path.abspath(args.list))
        names = []
        with open(args.list, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    names.append(json.loads(line)["filename"])
                if args.limit and len(names) >= args.limit:
                    break
        images = [os.path.join(img_dir, n) for n in names]
    else:
        images = collect_images(args)

    print("config  :", summary(cfg))
    cli_cfg = cfg.dump()                            # 记下命令行这版配置, 用来发现 --set 被 ckpt 覆盖
    model, cfg = load_model(ckpt, cfg, device)      # cfg 会被 ckpt 里记录的训练配置覆盖
    warn_cfg_drift(cli_cfg, cfg.dump())             # 结构参数不一致时明确提示, 而不是悄悄用 ckpt 的
    print("device  : %s | amp %s" % (device, args.amp))
    if not images:
        raise SystemExit("没有可处理的图片: --dir 下没有 %s 文件, 或 --list 里的 filename 都不存在"
                         % "/".join(IMG_EXTS))
    print("输入    : %d 张 | 输出 %s\n" % (len(images), out_dir))

    ok = fail = 0
    t0 = time.time()
    for p in images:
        if not os.path.exists(p):
            print("跳过 (不存在): %s" % p)
            fail += 1
            continue
        try:
            infer_one(model, cfg, device, p, out_dir, amp=args.amp,
                      save_viz=not args.no_viz, save_html=args.html)
            ok += 1
        except Exception as exc:                              # noqa: BLE001
            print("失败 %s -> %s" % (os.path.basename(p), exc))
            fail += 1

    dt = time.time() - t0
    print("\n完成: %d 张成功, %d 张失败, %.1fs (%.0f ms/张)"
          % (ok, fail, dt, 1000 * dt / max(ok + fail, 1)))
    print("结果目录: %s" % out_dir)


if __name__ == "__main__":
    main()
