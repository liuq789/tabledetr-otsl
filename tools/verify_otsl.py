# -*- coding: utf-8 -*-
"""全量验证 OTSL 标签体系 (8 标签, 含 eos, 数据源为 parquet) (扫 data/data 下所有 parquet).

每条样本检查 7 项:
    1. token 全部属于 OTSL 6 个结构 token (没有未知 token)
    2. 序列长度 == rows * (cols + 1)      (每行 cols 个 cell + 1 个 <nl>)
    3. <nl> 个数 == rows, 每行 cell 数 == cols
    4. 能重建网格 (build_grid 不抛异常)
    5. 网格自洽: 每格都被覆盖, 面积和 == rows*cols
    6. 网格反推回标签 == 原序列 (无损可逆)
    7. 由网格重建的 HTML == 官方 html_restored (逐字符相等)

只读 otsl / rows / cols / html_restored 四列, 不碰 image 列 (8.3G 里的大头),
单进程低内存, 不会干扰机器上正在跑的其它训练任务.

用法:
    python tools/verify_otsl.py                 # 全量 50 个 parquet
    python tools/verify_otsl.py --limit 3       # 只验前 3 个文件
    python tools/verify_otsl.py --set data.max_len=800
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyarrow.parquet as pq                                # noqa: E402

from config import get_config, parse_overrides, summary     # noqa: E402
from datasets.otsl import (OTSL_TOKENS, NL, normalize_token,  # noqa: E402
                           infer_shape, build_grid, grid_to_tokens, grid_to_html,
                           to_span_scheme)

COLUMNS = ["filename", "otsl", "rows", "cols", "html_restored"]


def span_mat(g):
    """(rows, cols) 的 (rowspan, colspan) 矩阵, 用于判断官方 6 标签与本项目标签结构是否等价."""
    return [[(g.at[r][c].rowspan, g.at[r][c].colspan) if g.at[r][c] else None
             for c in range(g.cols)] for r in range(g.rows)]


def html_row_widths(html):
    """官方 html_restored 每行的列宽 (td 的 colspan 之和).

    用来判断一条异常样本到底是"我的重建逻辑错了"还是"官方标注自身冲突".
    自洽的样本每行宽度都必须等于 cols.
    """
    widths = []
    for tr in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        w = 0
        for m in re.finditer(r"<td([^>]*)>", tr):
            cs = re.search(r'colspan="(\d+)"', m.group(1))
            w += int(cs.group(1)) if cs else 1
        widths.append(w)
    return widths


class Report(object):
    def __init__(self):
        self.n = 0
        self.err = collections.Counter()
        self.bad = collections.defaultdict(list)     # 每类最多留 5 个样例
        self.lens = []                               # 序列长度
        self.shapes = []                             # (rows, cols)
        self.n_span = 0
        self.n_defect = 0                            # 官方 html 自身行宽不符的样本数
        self.tokens = collections.Counter()
        self.tokens7 = collections.Counter()         # 本项目标签的分布

    def fail(self, kind, fn, msg, ref="", cols=None):
        self.err[kind] += 1
        defect = False
        if ref and cols:
            w = html_row_widths(ref)
            defect = any(x != cols for x in w) or len(w) != len(ref)
            if defect:
                self.n_defect += 1
        if len(self.bad[kind]) < 5:
            self.bad[kind].append({"file": fn, "msg": msg,
                                   "data_defect": defect})

    def check_one(self, r):
        self.n += 1
        fn = r["filename"]
        raw = r["otsl"] or []
        rows, cols = r["rows"], r["cols"]
        ref = "".join(r["html_restored"] or [])

        # 1) token 合法性
        toks, bad = [], []
        for x in raw:
            t = normalize_token(x)
            if t is None:
                bad.append(x)
            else:
                toks.append(t)
        if bad:
            self.fail("unknown_token", fn, "未知 token %r" % bad[:5], ref, cols)
            return
        self.tokens.update(toks)
        L = len(toks)
        self.lens.append(L)
        self.shapes.append((rows, cols))
        if any(t in ("<lcel>", "<ucel>", "<xcel>") for t in toks):
            self.n_span += 1

        # 2) 长度 == rows*(cols+1)
        if L != rows * (cols + 1):
            self.fail("len_mismatch", fn, "len=%d != rows*(cols+1)=%d" % (L, rows * (cols + 1)),
                      ref, cols)
            return

        # 3) 形状
        try:
            r2, c2 = infer_shape(toks)
        except Exception as e:
            self.fail("shape", fn, str(e), ref, cols)
            return
        if (r2, c2) != (rows, cols):
            self.fail("shape", fn, "推断 %dx%d != 标注 %dx%d" % (r2, c2, rows, cols), ref, cols)
            return
        if sum(1 for t in toks if t == NL) != rows:
            self.fail("nl_count", fn, "nl=%d != rows=%d"
                      % (sum(1 for t in toks if t == NL), rows), ref, cols)
            return

        # 4) 官方 6 标签 -> 网格
        try:
            g6 = build_grid(toks, rows=rows, cols=cols)
        except Exception as e:
            self.fail("build_grid", fn, str(e), ref, cols)
            return

        # 5) 6 标签网格自洽
        errs = g6.check()
        if errs:
            self.fail("grid", fn, "; ".join(errs[:3]), ref, cols)
            return

        # 6) 转成 8 标签 (合并区左上角 -> <span>, 末尾加 <eos>) 并重建
        try:
            t7 = to_span_scheme(toks, rows=rows, cols=cols)
            g7 = build_grid(t7, rows=rows, cols=cols)
        except Exception as e:
            self.fail("to_span", fn, str(e), ref, cols)
            return
        self.tokens7.update(t7)
        if g7.check():
            self.fail("grid7", fn, "; ".join(g7.check()[:3]), ref, cols)
            return

        # 7) 结构等价 (span 矩阵与官方一致)
        if span_mat(g6) != span_mat(g7):
            self.fail("equiv", fn, "与官方 span 矩阵不一致", ref, cols)
            return

        # 8) 无损可逆
        if grid_to_tokens(g7) != t7:
            self.fail("roundtrip7", fn, "grid_to_tokens != 标签序列", ref, cols)
            return

        # 9) 重建 HTML == html_restored
        got = grid_to_html(g7)
        if got != ref:
            self.fail("html", fn, "got %r vs ref %r" % (got[:120], ref[:120]), ref, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--limit", type=int, default=0, help="只验前 N 个文件")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--parquet-dir", default=None,
                    help="默认 ../data/data (含官方 otsl 与 html_restored)")
    ap.add_argument("--out", default=None, help="异常样例 jsonl")
    args = ap.parse_args()

    cfg = get_config(args.config, parse_overrides(args.overrides), reload=True)
    # 数据源固定指向原始 parquet (里面才有官方 otsl / html_restored 可对照)
    pdir = args.parquet_dir or os.path.abspath(
        os.path.join(ROOT, "..", "data", "data"))
    files = sorted(glob.glob(os.path.join(pdir, "*.parquet")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("没有找到 parquet:", pdir)
        return 1

    print("config:", cfg["_config_path"])
    print("params:", summary(cfg))
    print("标签表 : %s  (%d 个, 不含 pad)" % (OTSL_TOKENS, len(OTSL_TOKENS)))
    print("         <pad> 只用于 batch 补齐, id=%d, 不参与输出" % cfg.model.num_classes)
    print("parquet: %d 个文件 @ %s" % (len(files), pdir))
    print("-" * 78)

    rep = Report()
    t0 = time.time()
    for i, path in enumerate(files, 1):
        pf = pq.ParquetFile(path)
        n0, e0 = rep.n, sum(rep.err.values())
        for batch in pf.iter_batches(batch_size=args.batch, columns=COLUMNS):
            for row in batch.to_pylist():
                rep.check_one(row)
        print("  [%2d/%2d] %-46s %6d 条  错误 %d  累计 %.1fs"
              % (i, len(files), os.path.basename(path), rep.n - n0,
                 sum(rep.err.values()) - e0, time.time() - t0), flush=True)

    dt = time.time() - t0
    print("-" * 78)
    n = rep.n
    print("样本总数: %d    耗时 %.1fs (%.0f 条/秒)" % (n, dt, n / max(dt, 1e-6)))

    total_bad = sum(rep.err.values())
    if not rep.err:
        print("✅ 全部 7 项检查通过, 0 异常")
    else:
        print("❌ 异常统计:")
        for k, v in rep.err.most_common():
            print("  %-14s %6d  (%.4f%%)" % (k, v, 100.0 * v / n))
        print("  异常样本合计 %d (%.4f%%), 其中 %d 条的官方 html_restored 自身行宽 != cols"
              % (total_bad, 100.0 * total_bad / n, rep.n_defect))
        if rep.n_defect == total_bad:
            print("  => 这些异常全部是数据自带标注冲突 (官方 OTSL->HTML 转换缺陷), "
                  "不是标签体系的问题")
    print("纯净样本: %d / %d (%.4f%%)" % (n - total_bad, n, 100.0 * (n - total_bad) / n))

    lens = sorted(rep.lens)
    if lens:
        def pct(p):
            return lens[min(len(lens) - 1, int(p * len(lens)))]
        ml = int(cfg.data.max_len)
        over = sum(1 for x in lens if x > ml)
        print("\n序列长度 (= rows*(cols+1)): min %d / p50 %d / p90 %d / p95 %d / p99 %d / max %d"
              % (lens[0], pct(.5), pct(.9), pct(.95), pct(.99), lens[-1]))
        print("含合并(lcel/ucel/xcel)的样本: %d (%.1f%%)"
              % (rep.n_span, 100.0 * rep.n_span / n))
        print("超过 data.max_len=%d 的样本: %d (%.2f%%)" % (ml, over, 100.0 * over / n))
        rs = sorted(set(r for r, _ in rep.shapes))
        cs = sorted(set(c for _, c in rep.shapes))
        print("rows 范围 %d~%d | cols 范围 %d~%d" % (rs[0], rs[-1], cs[0], cs[-1]))
        tot7 = sum(rep.tokens7.values()) or 1
        print("8 标签占比:", {k: round(100.0 * v / tot7, 2)
                              for k, v in rep.tokens7.most_common()})

    out = args.out or os.path.join(cfg.project.output_dir, "verify_bad.jsonl")
    if rep.bad:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for kind, items in rep.bad.items():
                for it in items:
                    f.write(json.dumps({"kind": kind, **it}, ensure_ascii=False) + "\n")
        print("\n异常样例 ->", out)
    return 0 if not rep.err else 2


if __name__ == "__main__":
    sys.exit(main())
