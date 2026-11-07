# -*- coding: utf-8 -*-
"""OTSL 标签的肉眼验证: 原图 vs 用标签序列重建的空壳表格.

两种数据源:
    python tools/viz_otsl.py --jsonl ../tabeldetr/data/train_50k/PubTabNet_train.jsonl
        # 训练实际用的 jsonl + 同名 png, 现场转标签 (推荐)
    python tools/viz_otsl.py
        # 读 parquet (含官方 otsl 与 image 列), 用做标签体系验证
"""
import argparse
import base64
import glob
import html as H
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import get_config, parse_overrides                # noqa: E402
from datasets.convert import structure_to_otsl                # noqa: E402
from datasets.otsl import (build_grid, grid_to_html,          # noqa: E402
                           normalize_token, to_span_scheme)

TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>OTSL 标签重建验证</title>
<style>
 body{margin:0;padding:24px;background:#f6f7f9;color:#101828;
      font-family:Segoe UI,PingFang SC,Microsoft YaHei,sans-serif;font-size:13px}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:#667085;margin-bottom:18px;line-height:1.7}
 .sub code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:12px}
 .card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;margin-bottom:16px;overflow:hidden}
 .hd{display:flex;gap:10px;align-items:center;padding:9px 14px;border-bottom:1px solid #e4e7ec}
 .fn{font-family:Consolas,monospace;font-size:12px;color:#344054}
 .chip{font-size:11px;padding:2px 8px;border-radius:99px;background:#f2f4f7;color:#475467}
 .chip.span{background:#fef3e6;color:#b54708}
 .panes{display:grid;grid-template-columns:1fr 1fr}
 .p{padding:12px 14px;min-width:0;overflow:auto}
 .p+.p{border-left:1px solid #e4e7ec}
 .p img{max-width:100%;display:block;margin:0 auto}
 table{border-collapse:collapse;margin:0 auto;width:100%;table-layout:fixed}
 td{border:1px solid #98a2b3;height:var(--h,20px);min-height:var(--h,20px);padding:0 4px;font-size:11px;color:#98a2b3}
 td[colspan],td[rowspan]{background:#fff8ee;border-color:#e2a35c}
 .seq{font-family:Consolas,monospace;font-size:11px;line-height:1.9;padding:8px 14px;
      border-top:1px solid #e4e7ec;background:#fbfcfd;word-break:break-all}
 .t{display:inline-block;padding:0 3px;border-radius:3px;margin:1px}
 .t-f{background:#e6f7f4;color:#0b7565}.t-e{background:#f2f4f7;color:#667085}
 .t-l,.t-u,.t-x{background:#fef3e6;color:#b54708}.t-n{background:#e8effb;color:#1d4ed8}
 .t-s{background:#fde8f2;color:#a3165c;font-weight:600}
 .t-eos{background:#e9e9ef;color:#494f6b;font-weight:600}
 .legend{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 16px;font-size:12px;color:#475467}
</style></head><body>
<h1>OTSL 标签重建验证</h1>
<div class="sub">
8 个标签（<b>不含 pad</b>）：<code>fcel</code> 有内容 ·
<code>ecel</code> 空 · <code>span</code> 合并区左上角 · <code>lcel</code> 左合并 ·
<code>ucel</code> 上合并 · <code>xcel</code> 交叉 · <code>nl</code> 换行 · <code>eos</code> 序列结束（无 thead/tbody）<br>
左边是原图，右边是由这条样本的标签序列重建的空壳表格；橙色底是被 colspan/rowspan 覆盖的合并区。
</div>
<div class="legend">
<span><i class="t t-f">fcel</i> 有内容</span><span><i class="t t-e">ecel</i> 空</span>
<span><i class="t t-l">lcel</i> 横向延续</span><span><i class="t t-u">ucel</i> 纵向延续</span>
<span><i class="t t-x">xcel</i> 2D 内部</span><span><i class="t t-n">nl</i> 换行</span>
<span><i class="t t-s">span</i> 合并区左上角</span>
<span><i class="t t-eos">eos</i> 结束</span>
</div>
__CARDS__
</body></html>"""

_CLS = {"<fcel>": "t-f", "<ecel>": "t-e", "<span>": "t-s", "<lcel>": "t-l",
        "<ucel>": "t-u", "<xcel>": "t-x", "<nl>": "t-n", "<eos>": "t-eos"}


def render_card(filename, toks, rows, cols, img_bytes):
    """toks 是标签序列; 用渲染出来的表格对照原图."""
    g = build_grid(toks, rows=rows, cols=cols)
    errs = g.check()
    b64 = base64.b64encode(img_bytes or b"").decode()
    seq = "".join('<i class="t %s">%s</i>' % (_CLS.get(t, "t-e"), H.escape(t[1:-1]))
                  for t in toks)
    n_span = sum(1 for c in g.cells if c.rowspan > 1 or c.colspan > 1)
    cell_h = max(14, min(40, 760 // max(rows, 1)))        # 行多则压矮, 高度接近原图
    tbl = grid_to_html(g, wrap=True).replace(
        "<table>", '<table style="--h:%dpx">' % cell_h)
    warn = "" if not errs else \
        '<span class="chip span">网格异常: %s</span>' % H.escape(str(errs[:2]))
    return """<div class="card">
 <div class="hd"><span class="fn">%s</span>
   <span class="chip">%d 行 × %d 列</span>
   <span class="chip">%d 个 cell</span>
   <span class="chip%s">%d 个合并区</span>
   <span class="chip">%d token</span>
   %s
 </div>
 <div class="panes">
   <div class="p"><img src="data:image/png;base64,%s"></div>
   <div class="p">%s</div>
 </div>
 <div class="seq">%s</div>
</div>""" % (H.escape(filename), rows, cols, len(g.cells),
             " span" if n_span else "", n_span, len(toks), warn, b64, tbl, seq)


def from_jsonl(path, img_dir, n, span_only):
    """训练数据源: jsonl + 同名 png, structure 现场转标签."""
    cards, seen = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if seen > 40000:
                break
            d = json.loads(line)
            html = d.get("html") or {}
            toks, R, C, _, holes = structure_to_otsl(
                (html.get("structure") or {}).get("tokens") or [],
                html.get("cells") or [])
            if holes or not toks:
                continue
            if span_only and not any(t in ("<lcel>", "<ucel>", "<xcel>") for t in toks):
                continue
            try:
                with open(os.path.join(img_dir, d["filename"]), "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            cards.append(render_card(d["filename"], toks, R, C, data))
            if len(cards) >= n:
                break
    return cards


def from_parquet(parquet_dir, n, span_only):
    """标签体系数据源: parquet 自带官方 otsl + image 列."""
    import pyarrow.parquet as pq
    cards = []
    for path in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["filename", "otsl", "rows", "cols", "image"]):
            for r in batch.to_pylist():
                toks6 = [normalize_token(x) for x in (r["otsl"] or [])]
                if span_only and not any(t in ("<lcel>", "<ucel>", "<xcel>") for t in toks6):
                    continue
                try:
                    toks = to_span_scheme(toks6, rows=r["rows"], cols=r["cols"])
                except Exception:                        # noqa: BLE001
                    continue
                data = (r.get("image") or {}).get("bytes") or b""
                cards.append(render_card(r["filename"], toks, r["rows"], r["cols"], data))
                if len(cards) >= n:
                    return cards
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--jsonl", default=None,
                    help="从训练用的 jsonl + png 出图 (不给则读 parquet)")
    ap.add_argument("--img-dir", default=None, help="jsonl 模式下的图片目录")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--span-only", action="store_true", help="只要含合并单元格的")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = get_config(args.config, parse_overrides(args.overrides), reload=True)
    out = args.out or os.path.join(cfg.project.output_dir, "otsl_preview.html")

    if args.jsonl:
        jp = os.path.abspath(args.jsonl)
        img_dir = args.img_dir or os.path.dirname(jp)
        cards = from_jsonl(jp, img_dir, args.n, args.span_only)
        src = "jsonl: %s" % jp
    else:
        pdir = os.path.abspath(os.path.join(ROOT, "..", "data", "data"))
        cards = from_parquet(pdir, args.n, args.span_only)
        src = "parquet: %s" % pdir

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("__CARDS__", "\n".join(cards)))
    print("source :", src)
    print("samples:", len(cards))
    print("out    :", out)


if __name__ == "__main__":
    main()
