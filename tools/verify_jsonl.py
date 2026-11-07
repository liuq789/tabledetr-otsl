# -*- coding: utf-8 -*-
"""验证 jsonl 数据的 structure -> 网格标签转换能否还原出正确的表格结构.

训练实际读的是 tabeldetr/data/*.jsonl (只有 structure tokens + cells, 没有 otsl).
这里拿 parquet 里官方的 html_restored 当标准答案, 按 filename 关联, 逐字符比对:

    jsonl.structure.tokens + cells
        -> structure_to_otsl()      转换
        -> build_grid()             重建网格
        -> grid_to_html()           渲染
        == parquet.html_restored ?  官方答案

同时检查 5 项自洽性:
    1. holes == 0                        结构里没有空洞
    2. len(tokens) == rows*(cols+1)      长度与网格严格对应
    3. grid.check() 通过                  每格都被覆盖, 面积和 = rows*cols
    4. grid_to_tokens(grid) == tokens    可无损反推
    5. 网格起点数 == len(cells)           bbox 能一一挂上

用法:
    python tools/verify_jsonl.py                     # 默认验 config 里的 train_jsonl
    python tools/verify_jsonl.py --limit 5000        # 只验前 5000 条
    python tools/verify_jsonl.py --jsonl ../tabeldetr/data/train_200k/PubTabNet_train.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyarrow.parquet as pq                                    # noqa: E402

from config import get_config, parse_overrides                  # noqa: E402
from datasets.convert import structure_to_otsl                  # noqa: E402
from datasets.otsl import (build_grid, grid_to_html, grid_to_tokens,  # noqa: E402
                           normalize_token, CELL_TOKENS)


def load_reference(parquet_dir, need, log_every=10):
    """扫 parquet 建 filename -> html_restored 的对照表 (只读两列, 不碰 image)."""
    ref = {}
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    t0 = time.time()
    for i, path in enumerate(files, 1):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048,
                                     columns=["filename", "html_restored", "otsl", "rows", "cols"]):
            for r in batch.to_pylist():
                fn = r["filename"]
                if fn in need:
                    ref[fn] = {"html": "".join(r["html_restored"] or []),
                               "otsl": r["otsl"], "rows": r["rows"], "cols": r["cols"]}
        if i % log_every == 0 or i == len(files):
            print("  parquet [%2d/%2d] 已匹配 %d 条  累计 %.1fs"
                  % (i, len(files), len(ref), time.time() - t0), flush=True)
    return ref


def official_is_broken(info):
    """官方 otsl 自己建不出网格 -> 是数据缺陷, 不是我们转换的问题."""
    try:
        toks = [normalize_token(x) for x in (info.get("otsl") or [])]
        g = build_grid(toks, rows=info.get("rows"), cols=info.get("cols"))
        return bool(g.check())
    except Exception:                                    # noqa: BLE001
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--jsonl", default=None, help="默认取 config 的 train_jsonl")
    ap.add_argument("--limit", type=int, default=0, help="只验前 N 条")
    ap.add_argument("--parquet-dir", default=None,
                    help="默认取 ../data/data (含官方 html_restored)")
    args = ap.parse_args()

    cfg = get_config(args.config, parse_overrides(args.overrides), reload=True)
    jsonl = args.jsonl or cfg.paths.train_jsonl
    pdir = args.parquet_dir or os.path.abspath(os.path.join(ROOT, "..", "data", "data"))

    print("jsonl  :", jsonl)
    print("参考   :", pdir)

    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print("样本   :", len(rows))

    need = {r["filename"] for r in rows}
    print("建对照表 (只需要 %d 个 filename)..." % len(need))
    ref = load_reference(pdir, need)
    print("匹配到官方 HTML: %d / %d" % (len(ref), len(need)))
    print("-" * 74)

    stat = {"n": 0, "no_ref": 0, "err": 0, "html_diff": 0, "official_defect": 0,
            "holes": 0, "len_bad": 0, "grid_bad": 0, "rt_bad": 0, "cells_bad": 0}
    examples = []
    t0 = time.time()

    for r in rows:
        stat["n"] += 1
        fn = r["filename"]
        html = r.get("html") or {}
        tokens = (html.get("structure") or {}).get("tokens") or []
        cells = html.get("cells") or []
        answer = ref.get(fn)
        if answer is None:
            stat["no_ref"] += 1
            continue
        try:
            toks, R, C, orders, holes = structure_to_otsl(tokens, cells)
        except Exception as e:
            stat["err"] += 1
            if len(examples) < 5:
                examples.append((fn, "转换异常: %s" % e))
            continue
        if holes:
            stat["holes"] += 1
            if len(examples) < 5:
                examples.append((fn, "结构有 %d 个空洞" % holes))
            continue
        if not toks or len(toks) != R * (C + 1) + 1:
            stat["len_bad"] += 1
            if len(examples) < 5:
                examples.append((fn, "长度 %d != %d*(%d+1)+1" % (len(toks), R, C)))
            continue
        try:
            g = build_grid(toks, rows=R, cols=C)
        except Exception as e:
            stat["grid_bad"] += 1
            if len(examples) < 5:
                examples.append((fn, "重建失败: %s" % e))
            continue
        if g.check():
            stat["grid_bad"] += 1
            if len(examples) < 5:
                examples.append((fn, "网格不自洽: %s" % g.check()[:2]))
            continue
        if grid_to_tokens(g) != toks:
            stat["rt_bad"] += 1
            if len(examples) < 5:
                examples.append((fn, "反推不一致"))
            continue
        n_origin = sum(1 for t in toks if t in CELL_TOKENS)
        if n_origin != len(cells):
            stat["cells_bad"] += 1
            if len(examples) < 5:
                examples.append((fn, "起点 %d != cells %d" % (n_origin, len(cells))))
            continue
        got = grid_to_html(g)
        if got != answer["html"]:
            if official_is_broken(answer):
                stat["official_defect"] += 1          # 官方标注自身残缺, 我们的转换以 structure 为准
                continue
            stat["html_diff"] += 1
            if len(examples) < 5:
                examples.append((fn, "HTML 不同\n      got: %s\n      ref: %s"
                                 % (got[:150], answer["html"][:150])))

    dt = time.time() - t0
    n = stat["n"]
    print("-" * 74)
    print("样本 %d | 耗时 %.1fs (%.0f 条/秒)" % (n, dt, n / max(dt, 1e-6)))
    print("无官方答案        : %d" % stat["no_ref"])
    print("转换异常          : %d" % stat["err"])
    print("结构有空洞        : %d" % stat["holes"])
    print("长度不符          : %d" % stat["len_bad"])
    print("网格不自洽        : %d" % stat["grid_bad"])
    print("反推不一致        : %d" % stat["rt_bad"])
    print("起点数 != cells 数 : %d" % stat["cells_bad"])
    print("HTML 与官方不同   : %d" % stat["html_diff"])
    print("官方标注自身残缺   : %d  (官方 otsl 建不出网格, 已按 structure 为准)" % stat["official_defect"])
    checked = n - stat["no_ref"]
    ok_bad = sum(v for k, v in stat.items() if k not in ("n", "no_ref", "official_defect"))
    print("\n通过 %d / %d (%.4f%%)"
          % (checked - ok_bad, checked, 100.0 * (checked - ok_bad) / max(checked, 1)))
    if examples:
        print("\n异常样例:")
        for fn, msg in examples:
            print("  -", fn, msg)
    return 0 if ok_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
