# -*- coding: utf-8 -*-
"""OTSL 8 标签表格结构 (不含 thead/tbody, 纯网格).

8 个标签 (分类头 8 维, pad 不在其中)
    0 <fcel>  有内容、非合并的单元格
    1 <ecel>  空、非合并的单元格
    2 <span>  合并区左上角 (带 colspan 或 rowspan, 内容有无不限)
    3 <lcel>  被左边单元格横向跨过 (colspan 的延续位)
    4 <ucel>  被上边单元格纵向跨过 (rowspan 的延续位)
    5 <xcel>  被左、上同时跨过 (2D 合并区的内部)
    6 <nl>    行结束

    7 <eos>   序列结束 (推理时遇到它就停)

    <pad> 不在标签表里: 它只是 batch 补齐用的占位 id (=8), 不参与 loss 也不参与评估.

为什么这 7 个就够
    原 TableMaster 方案 39 个 token 是因为把 colspan="2".."10" / rowspan="2".."10"
    和 11 种 <eb*> 形态全枚举了. OTSL 改用"网格 + 相对位置"表达:
    span 的尺寸由 <span> 起点 + lcel/ucel/xcel 的分布共同给出,
    行数 = <nl> 个数, 列数 = 每行 token 数, 所以不需要任何带数字的参数 token.

    比官方 6 标签多出的 <span> 是"合并区左上角"的显式标记:
    模型输出到这一格就知道要带 rowspan/colspan, 不必等后面出现 lcel/ucel 才反推.
    官方 6 标签只有 fcel/ecel, 合并起点和普通格在标签上无法区分.

序列形状
    长度固定 = rows * cols + rows  (每行 cols 个 cell + 1 个 <nl>)
    与 rows/cols 强绑定, 可直接 shape 成 (rows, cols+1).

用法
    from datasets.otsl import OTSL_TOKENS, parse_otsl, build_grid, grid_to_html
    toks6 = parse_otsl("fcel lcel nl ucel xcel nl")   # 官方 6 标签
    toks7 = to_span_scheme(toks6)                     # 合并起点 -> <span>, 末尾补 <eos>
    g = build_grid(toks7)
    html = grid_to_html(g)
"""
import os
import sys

from config import get_config

CFG = get_config()

# ---------------------------------------------------------------- 词表

FCEL, ECEL, SPAN, LCEL, UCEL, XCEL, NL, EOS = (
    "<fcel>", "<ecel>", "<span>", "<lcel>", "<ucel>", "<xcel>", "<nl>", "<eos>")
PAD = "<pad>"                            # 仅用于兼容外部 otsl 字符串的解析, 不是类别, 无 id

# 8 个结构标签 (不含 pad) -> 分类输出层 8 维
#   前 7 个表达网格, 末尾 <eos> 让模型自己学"序列到哪结束"
#   (OTSL 是定长网格 rows*(cols+1), 没有 eos 时推理无法判断有效长度)
OTSL_TOKENS = [FCEL, ECEL, SPAN, LCEL, UCEL, XCEL, NL, EOS]

STOI = {t: i for i, t in enumerate(OTSL_TOKENS)}
ITOS = list(OTSL_TOKENS)
NUM_CLASSES = len(OTSL_TOKENS)          # 8 = 全部真实类别, 不含 pad
IGNORE_INDEX = -100                     # 补齐位唯一的哨兵: labels 填它, cross_entropy 跳过
CELL_TOKENS = (FCEL, ECEL, SPAN)        # 会开启新单元格
OCCUPY_TOKENS = (LCEL, UCEL, XCEL)      # 被合并覆盖的格子

# 官方 otsl 字符串里可能是 "fcel" / "<fcel>" / "fcel," 几种写法, 统一成带尖括号
_ALIAS = {}
for _t in OTSL_TOKENS:
    _n = _t[1:-1]                       # fcel
    _ALIAS[_n] = _t
    _ALIAS[_t] = _t
_ALIAS["span"] = SPAN
_ALIAS["<span>"] = SPAN


def normalize_token(tok):
    """'fcel' / '<fcel>' / 'fcel,' -> '<fcel>'; 不认识的返回 None."""
    t = tok.strip().strip(",;").strip()
    if not t:
        return None
    if t in _ALIAS:
        return _ALIAS[t]
    low = t.lower()
    if low in _ALIAS:
        return _ALIAS[low]
    if "<" in t:                        # <fcel> 之类, 去掉尖括号再查
        core = t.strip("<>").lower()
        return _ALIAS.get("<%s>" % core)
    return None


def parse_otsl(text):
    """官方 otsl 字符串 -> 标签序列. 未知 token 抛 ValueError (便于发现格式差异)."""
    if text is None:
        return []
    out, bad = [], []
    for raw in text.replace("\n", " ").split():
        t = normalize_token(raw)
        if t is None:
            bad.append(raw)
        else:
            out.append(t)
    if bad:
        raise ValueError("otsl 里出现未知 token: %r" % sorted(set(bad))[:10])
    return out


def encode(tokens):
    return [STOI[t] for t in tokens]


def decode(ids):
    return [ITOS[i] for i in ids if 0 <= i < NUM_CLASSES]


# ---------------------------------------------------------------- 网格重建

class Cell(object):
    __slots__ = ("r", "c", "rowspan", "colspan", "has_content", "is_span", "text")

    def __init__(self, r, c, has_content, is_span=False):
        self.r, self.c = r, c
        self.rowspan = self.colspan = 1
        self.has_content = has_content
        self.is_span = is_span          # 是不是合并区左上角 (<span>)
        self.text = ""

    @property
    def area(self):
        return self.rowspan * self.colspan

    def __repr__(self):
        kind = "span" if (self.is_span or self.area > 1) else ("fcel" if self.has_content else "ecel")
        return "Cell(r=%d,c=%d,rs=%d,cs=%d,%s)" % (
            self.r, self.c, self.rowspan, self.colspan, kind)


class Grid(object):
    """重建结果: 行数/列数/单元格表/占据矩阵."""

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.cells = []                 # 所有 origin cell
        self.at = [[None] * cols for _ in range(rows)]   # 每格属于哪个 cell
        self.occupied = [[False] * cols for _ in range(rows)]

    def add(self, cell):
        self.cells.append(cell)
        self.at[cell.r][cell.c] = cell

    def fill_cover(self):
        """按 span 把 cell 铺满它覆盖的区域."""
        for cell in self.cells:
            for r in range(cell.r, cell.r + cell.rowspan):
                for c in range(cell.c, cell.c + cell.colspan):
                    if (r, c) != (cell.r, cell.c):
                        self.at[r][c] = cell
                        self.occupied[r][c] = True

    def check(self):
        """一致性: 每个位置都被某个 cell 覆盖, 且面积之和 == rows*cols."""
        errs = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.at[r][c] is None:
                    errs.append("空位 (%d,%d)" % (r, c))
        area = sum(x.area for x in self.cells)
        if area != self.rows * self.cols:
            errs.append("面积和 %d != rows*cols %d" % (area, self.rows * self.cols))
        return errs

    def spans(self):
        """(rows, cols) 的 span 矩阵, 每格 (rowspan, colspan, has_content) —— 用于比对."""
        mat = [[None] * self.cols for _ in range(self.rows)]
        for cell in self.cells:
            mat[cell.r][cell.c] = (cell.rowspan, cell.colspan, cell.has_content)
        return mat


def infer_shape(tokens, nl_token=NL):
    """按 <nl> 推断 (rows, cols), 并校验每行 token 数一致."""
    rows, cur = [], 0
    for t in tokens:
        if t == EOS:                    # 标签序列末尾的结束符, 不算格子
            break
        if t == nl_token:
            rows.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        raise ValueError("序列末尾缺少 %s (最后一行 %d 个 cell)" % (nl_token, cur))
    if not rows:
        return 0, 0
    if len(set(rows)) != 1:
        raise ValueError("各行 cell 数不一致: %r" % sorted(set(rows)))
    return len(rows), rows[0]


def build_grid(tokens, rows=None, cols=None, strict=True):
    """标签序列 -> Grid.

    lcel/ucel/xcel 的归属规则 (与 OTSL 论文一致):
        lcel -> 同行左侧最近的 origin cell (fcel/ecel)
        ucel -> 向上第一个 origin cell (跨过连续的 ucel/xcel)
        xcel -> 已被 lcel 加宽 / ucel 加高 的合并区内部, 只标记占据
    """
    tok_rows = []
    cur = []
    for t in tokens:
        if t == PAD or t == EOS:        # 补齐位 / 结束符: 序列到此为止
            break
        if t == NL:
            tok_rows.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        raise ValueError("序列末尾缺少 <nl>")

    r_num, c_num = len(tok_rows), (len(tok_rows[0]) if tok_rows else 0)
    if any(len(x) != c_num for x in tok_rows):
        raise ValueError("各行长度不一致: %r" % [len(x) for x in tok_rows])
    if rows is not None and int(rows) != r_num:
        raise ValueError("rows 不符: 标注 %s vs 序列 %d" % (rows, r_num))
    if cols is not None and int(cols) != c_num:
        raise ValueError("cols 不符: 标注 %s vs 序列 %d" % (cols, c_num))

    g = Grid(r_num, c_num)
    for r, line in enumerate(tok_rows):
        for c, t in enumerate(line):
            if t in CELL_TOKENS:
                g.add(Cell(r, c, t == FCEL, t == SPAN))
            elif t == LCEL:
                origin = _left_origin(g, r, c)
                if origin is None:
                    if strict:
                        raise ValueError("(%d,%d) lcel 左边没有 origin" % (r, c))
                    g.add(Cell(r, c, False))             # 兜底: 当作新空单元格
                    continue
                origin.colspan = max(origin.colspan, c - origin.c + 1)
                g.at[r][c] = origin                      # 立刻记归属, 后面 xcel 要靠它
            elif t == UCEL:
                origin = _up_origin(g, r, c)
                if origin is None:
                    if strict:
                        raise ValueError("(%d,%d) ucel 上面没有 origin" % (r, c))
                    g.add(Cell(r, c, False))             # 兜底
                    continue
                origin.rowspan = max(origin.rowspan, r - origin.r + 1)
                g.at[r][c] = origin
            elif t == XCEL:
                origin = _left_cover(g, r, c) or _up_cover(g, r, c)
                if origin is None:
                    # 极少数标注冲突的样本: xcel 既无左也无上, 严格模式直接报错,
                    # 宽松模式补一个新格, 保证网格自洽 (这类样本由 dataset 统计并跳过)
                    if strict:
                        raise ValueError("(%d,%d) xcel 既无左侧也无上方 origin" % (r, c))
                    g.add(Cell(r, c, False))
                    continue
                g.at[r][c] = origin
            else:
                raise ValueError("未知 token %r" % (t,))
    g.fill_cover()
    return g


def _left_origin(g, r, c):
    """(r,c) 左边最近的 cell (lcel 要往它身上加 colspan, 所以不要求已覆盖)."""
    for cc in range(c - 1, -1, -1):
        cell = g.at[r][cc]
        if cell is not None:
            return cell
    return None


def _up_origin(g, r, c):
    for rr in range(r - 1, -1, -1):
        cell = g.at[rr][c]
        if cell is not None:
            return cell
    return None


def _left_cover(g, r, c):
    """(r,c) 左边最近的、且 span 真的覆盖到 c 的 cell (xcel 归属用)."""
    for cc in range(c - 1, -1, -1):
        cell = g.at[r][cc]
        if cell is not None:
            return cell if cell.c + cell.colspan > c else None
    return None


def _up_cover(g, r, c):
    for rr in range(r - 1, -1, -1):
        cell = g.at[rr][c]
        if cell is not None:
            return cell if cell.r + cell.rowspan > r else None
    return None


def grid_to_html(g, text_of=None, wrap=False):
    """Grid -> 结构化 HTML (只含 tr/td + colspan/rowspan, 无 thead/tbody).

    属性顺序与 PubTabNet-OTSL 的 html_restored 一致: colspan 在前, rowspan 在后.
    wrap=True 时外套 <table> (直接当 HTML 用时).
    """
    parts = ["<table>"] if wrap else []
    for r in range(g.rows):
        parts.append("<tr>")
        for c in range(g.cols):
            if g.occupied[r][c]:
                continue
            cell = g.at[r][c]
            if cell is None:                     # 宽松模式的兜底, 不让渲染崩掉
                parts.append("<td></td>")
                continue
            attrs = ""
            if cell.colspan > 1:
                attrs += ' colspan="%d"' % cell.colspan
            if cell.rowspan > 1:
                attrs += ' rowspan="%d"' % cell.rowspan
            text = ""
            if text_of is not None:
                text = text_of(cell) or ""
            parts.append("<td%s>%s</td>" % (attrs, text))
        parts.append("</tr>")
    if wrap:
        parts.append("</table>")
    return "".join(parts)


def grid_to_tokens(g):
    """Grid -> 标签序列 (7 个网格标签 + <eos>, 合并区左上角输出 <span>).

    反向验证: tokens -> grid -> tokens 应完全一致.
    """
    out = []
    for r in range(g.rows):
        line = []
        for c in range(g.cols):
            if g.occupied[r][c]:
                owner = g.at[r][c]
                if owner.r == r:
                    line.append(LCEL)
                elif owner.c == c:
                    line.append(UCEL)
                else:
                    line.append(XCEL)
            else:
                cell = g.at[r][c]
                if cell.is_span or cell.area > 1:
                    line.append(SPAN)          # 合并区左上角
                else:
                    line.append(FCEL if cell.has_content else ECEL)
        out.extend(line)
        out.append(NL)
    out.append(EOS)                     # 结束符
    return out


def grid_to_six_tokens(g):
    """Grid -> 官方 6 标签序列 (不区分合并起点), 用于和数据集原始 otsl 比对."""
    out = []
    for r in range(g.rows):
        line = []
        for c in range(g.cols):
            if g.occupied[r][c]:
                owner = g.at[r][c]
                if owner.r == r:
                    line.append(LCEL)
                elif owner.c == c:
                    line.append(UCEL)
                else:
                    line.append(XCEL)
            else:
                line.append(FCEL if g.at[r][c].has_content else ECEL)
        out.extend(line)
        out.append(NL)
    return out


def to_span_scheme(tokens6, rows=None, cols=None):
    """官方 6 标签 -> 本项目标签 (6 个官方 + <span> 标记合并区左上角, 末尾补 <eos>).

    这是唯一需要从原始数据做的一步转换: 解析出网格后, 把带 colspan/rowspan
    的起点格改标成 <span>, 其余保持原样. 结构信息零损失 (span 尺寸仍由
    lcel/ucel/xcel 的分布给出).
    """
    g = build_grid(tokens6, rows=rows, cols=cols)
    out = []
    for r in range(g.rows):
        for c in range(g.cols):
            if g.occupied[r][c]:
                owner = g.at[r][c]
                if owner.r == r:
                    out.append(LCEL)
                elif owner.c == c:
                    out.append(UCEL)
                else:
                    out.append(XCEL)
            else:
                cell = g.at[r][c]
                out.append(SPAN if cell.area > 1
                           else (FCEL if cell.has_content else ECEL))
        out.append(NL)
    out.append(EOS)                     # 与训练标签一致: 末尾带结束符
    return out


# ---------------------------------------------------------------- 序列 <-> 张量

def to_fixed_matrix(tokens, rows=None, cols=None):
    """标签 -> (rows, cols+1) 的 id 矩阵, 末列是 <nl>. 便于 CNN/Transformer 直接回归."""
    g_rows, g_cols = infer_shape(tokens)
    if rows is not None and int(rows) != g_rows:
        raise ValueError("rows 不符")
    if cols is not None and int(cols) != g_cols:
        raise ValueError("cols 不符")
    ids = encode(tokens)
    mat = []
    k = 0
    for _ in range(g_rows):
        mat.append(ids[k:k + g_cols + 1])       # cols 个 cell + 1 个 nl (不含 eos)
        k += g_cols + 1
    return mat


__all__ = [
    "FCEL", "ECEL", "SPAN", "LCEL", "UCEL", "XCEL", "NL", "EOS", "PAD",
    "OTSL_TOKENS", "STOI", "ITOS", "NUM_CLASSES", "IGNORE_INDEX",
    "CELL_TOKENS", "OCCUPY_TOKENS", "normalize_token", "parse_otsl",
    "encode", "decode", "Cell", "Grid", "infer_shape", "build_grid",
    "grid_to_html", "grid_to_tokens", "grid_to_six_tokens", "to_span_scheme",
    "to_fixed_matrix", "CFG",
]
