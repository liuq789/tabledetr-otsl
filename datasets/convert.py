# -*- coding: utf-8 -*-
"""PubTabNet structure tokens -> 8 标签 OTSL (含 eos).

训练数据沿用 tabeldetr 那套 (jsonl + 同名 png), 标注里只有
`html.structure.tokens` 和 `cells[]`, 没有 otsl 字段.
这里在现场把 structure 转成 7 个网格标签的序列 (末尾补 <eos>), 避免再生成第二套标注文件.

PubTabNet structure tokens 的形态
    <thead> <tbody> </thead> </tbody> <tr> </tr> <td> </td>
    带属性时拆成三 token:  '<td'  +  ' colspan="4"'  +  '>'   (还可能带 rowspan)

转换步骤
    1. 扫 token, 按 <tr>...</tr> 分出每一行的 (colspan, rowspan)
    2. 按行推进算网格位置: 跳过被上面 rowspan 占住的格子
    3. 每格判标签: 起点 -> <span>(合并) / <fcel>(有字) / <ecel>(空)
                  其余 -> <lcel>(左边开的) / <ucel>(上边开的) / <xcel>(2D 内部)
    4. 行尾补 <nl>
"""
import re

from .otsl import FCEL, ECEL, SPAN, LCEL, UCEL, XCEL, NL, EOS, CELL_TOKENS

_ATTR_RE = re.compile(r'^\s*(colspan|rowspan)\s*=\s*"?(\d+)"?\s*$', re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def cell_has_content(cell):
    """判断一个 cell 有没有字. 三种数据形态都要兼容:

    * parquet 原始: bbox 是 5 元 [x0,y0,x1,y1,flag], flag=2 非空 / 1 空 (空格也有 bbox)
    * tabeldetr jsonl: bbox 是 4 元, 空 cell 干脆没有 bbox
    * 兜底: 看 tokens 去掉格式标签后还有没有字
    """
    bbox = cell.get("bbox")
    if bbox:
        if len(bbox) >= 5:
            return int(bbox[4]) == 2
        return True
    return len(_TAG_RE.sub("", "".join(cell.get("tokens") or [])).strip()) > 0
_TD_OPEN = ("<td>", "<td", "<th>", "<th")


def _parse_rows(tokens):
    """structure tokens -> [[{colspan, rowspan}, ...], ...] (每行一个 list)."""
    rows, cur = [], []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t == "</tr>":
            rows.append(cur)
            cur = []
            i += 1
        elif t in _TD_OPEN:
            colspan = rowspan = 1
            if t in ("<td", "<th"):                 # 属性在后续 token 里, 直到 '>'
                i += 1
                while i < n and tokens[i] != ">":
                    m = _ATTR_RE.match(tokens[i])
                    if m:
                        if m.group(1).lower() == "colspan":
                            colspan = int(m.group(2))
                        else:
                            rowspan = int(m.group(2))
                    i += 1
                i += 1
            else:
                i += 1
            cur.append({"colspan": max(1, colspan), "rowspan": max(1, rowspan)})
        else:
            i += 1
    if cur:
        rows.append(cur)
    return rows


def structure_to_grid(tokens):
    """structure tokens -> (rows, cols, grid_cells).

    grid_cells: [(r, c, rowspan, colspan, order), ...], 按行主序,
    order 是该 cell 在 PubTabNet cells[] 里的下标, 用来把 bbox 对齐到 token.
    """
    lines = _parse_rows(tokens)
    R = len(lines)
    if R == 0:
        return 0, 0, []

    taken = set()                       # 被上方 rowspan 占住的 (r, c)
    grid_cells = []
    order = 0
    max_c = 0
    for r, line in enumerate(lines):
        c = 0
        for spec in line:
            while (r, c) in taken:      # 跳过被占的列
                c += 1
            grid_cells.append((r, c, spec["rowspan"], spec["colspan"], order))
            order += 1
            for dr in range(spec["rowspan"]):
                for dc in range(spec["colspan"]):
                    if dr or dc:
                        taken.add((r + dr, c + dc))
            c += spec["colspan"]
        max_c = max(max_c, c)
    return R, max_c, grid_cells


def grid_to_otsl_tokens(R, C, grid_cells, has_content=None):
    """(rows, cols, grid_cells) -> (tokens7, order_of_token, holes).

    order_of_token 与 tokens 等长: 起点位给出 cells 下标, 其余为 -1.
    holes 是结构里没有被任何 cell 覆盖的格子数 (正常数据应为 0).
    """
    if R <= 0 or C <= 0:
        return [], [], 0

    owner = [[None] * C for _ in range(R)]
    for cell in grid_cells:
        r, c, rs, cs, order = cell
        for dr in range(rs):
            for dc in range(cs):
                rr, cc = r + dr, c + dc
                if rr < R and cc < C:
                    owner[rr][cc] = cell

    out, orders = [], []
    holes = 0
    for r in range(R):
        for c in range(C):
            cell = owner[r][c]
            if cell is None:                        # 结构有洞: 记数, 按空格填, 保持长度
                holes += 1
                out.append(ECEL)
                orders.append(-1)
                continue
            cr, cc, rs, cs, order = cell
            if cr == r and cc == c:                 # 起点
                if rs > 1 or cs > 1:
                    out.append(SPAN)
                else:
                    has = True if has_content is None else bool(has_content[order])
                    out.append(FCEL if has else ECEL)
                orders.append(order)
            elif cr == r:
                out.append(LCEL)
                orders.append(-1)
            elif cc == c:
                out.append(UCEL)
                orders.append(-1)
            else:
                out.append(XCEL)
                orders.append(-1)
        out.append(NL)
        orders.append(-1)
    return out, orders, holes


def structure_to_otsl(tokens, cells=None):
    """一步到位: structure tokens (+ PubTabNet cells) -> (tokens7, rows, cols, orders, holes).

    cells 用来判断起点格是"有内容"还是"空" (有 bbox 或 有 tokens 即算有内容).
    长度与网格起点数对不上时退回"全部按有内容", 只影响 fcel/ecel, 不影响结构.
    """
    R, C, grid_cells = structure_to_grid(tokens)

    has_content = None
    if cells and len(cells) == len(grid_cells):
        has_content = [cell_has_content(c) for c in cells]

    toks, orders, holes = grid_to_otsl_tokens(R, C, grid_cells, has_content)
    if toks:                            # 末尾加结束符, 长度 = rows*(cols+1)+1
        toks.append(EOS)
        orders.append(-1)
    return toks, R, C, orders, holes


__all__ = ["cell_has_content", "structure_to_grid", "grid_to_otsl_tokens",
           "structure_to_otsl"]
