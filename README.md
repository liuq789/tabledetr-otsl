# tabeldetr-otsl

表格结构识别（TSR）工程。输入一张表格图片，输出**结构标签序列 + 每个单元格的 bbox**，
由标签序列重建整张表的网格（含 rowspan / colspan）。

模型是 DETR 式**非自回归**解码：ResNet backbone 出特征序列，N 个可学习 query 并行输出整条标签序列，
一次前向出全部结果，不做 teacher forcing，也不递归解码。

```
图片 896x896 -> ResNet34(/32, 28x28=784 token) -> +2D 正弦位置编码
             -> 500 个可学习 query + 共享层 2 层 -> 分叉
                ├─ cls 支路 2 层 -> 500 x 8  (7 网格标签 + eos)
                └─ box 支路 3 层 -> 500 x 4  (cell 框, 归一化 0~1)
```

参数量 51.50M，输出 `(500, 8)` 标签 + `(500, 4)` 框。

---

## 1. 标签体系（8 类）

| id | 标签 | 含义 |
|---|---|---|
| 0 | `<fcel>` | 有内容的单元格 |
| 1 | `<ecel>` | 空单元格 |
| 2 | `<span>` | 合并区左上角（带 rowspan / colspan） |
| 3 | `<lcel>` | 被左边单元格横向跨过 |
| 4 | `<ucel>` | 被上边单元格纵向跨过 |
| 5 | `<xcel>` | 被左、上同时跨过（2D 合并区内部） |
| 6 | `<nl>` | 行结束 |
| 7 | `<eos>` | 序列结束 |

- 没有 thead / tbody，也没有 `colspan="2"` 这类带数字的参数 token。行数 = `<nl>` 个数，
  列数 = 每行 cell 数，合并信息由 `span` + `lcel/ucel/xcel` 的分布隐式表达。
- 序列长度 = `rows × (cols + 1) + 1`，与网格严格对齐，可直接 reshape 成 `(rows, cols+1)`。
- `<eos>` 让模型学「序列到哪结束」，推理时遇到它就停，不需要额外预测行列数。
- **pad 不是类别**：补齐位在 labels 里写 `-100`，`cross_entropy(ignore_index=-100)` 跳过，
  所以 `num_classes = 8` 就是全部真实类别，不占额外维度。
- 标签**不含文字**。需要文字时按输出的 bbox 裁图，再走 OCR。

数据侧统计（50k 子集）：序列长度 min 4 / 中位 60 / p95 231 / max 1300，约 30% 含合并单元格。

---

## 2. 数据格式

`jsonl` 标注 + 同目录同名图片：

```json
{
  "filename": "PMC1234_001_00.png",
  "split": "train",
  "imgid": 12,
  "html": {
    "structure": { "tokens": ["<thead>", "<tr>", "<td", " colspan=\"2\"", ">", "</td>", "..."] },
    "cells": [
      { "tokens": ["A", "B"], "bbox": [12, 34, 56, 48] },
      { "tokens": [] }
    ]
  }
}
```

- `structure.tokens`：表格结构骨架（`<td>` 与 `</td>` 分开写，带属性时拆成 `<td` + ` colspan="2"` + `>` 三个 token）
- `cells[i].bbox`：原图像素坐标，**空 cell 没有 bbox 字段**
- 第 i 个 `<td` 对应 `cells[i]`，一一对应

`datasets/convert.py` 在 `__getitem__` 里把 structure **现场转成**标签序列
（按 `<tr>` 分行 → 按 rowspan 推进算网格 → 每格判 fcel/ecel/span/lcel/ucel/xcel → 行尾 `nl` → 末尾 `eos`），
不生成第二套标注文件。数据路径由 `config.yaml` 的 `paths.train_jsonl / val_jsonl` 指定，
`paths.img_dir` 为空时取 jsonl 所在目录。

---

## 3. 目录结构

```
tabeldetr-otsl/
├── config.yaml              # 全部参数（数据 / 模型 / 训练），唯一来源
├── config.py                # 配置加载: get_config / parse_overrides / save_config / validate
├── datasets/
│   ├── otsl.py              # 词表、build_grid 网格重建、grid_to_tokens 反推、grid_to_html
│   ├── convert.py           # structure tokens -> 标签序列
│   └── jsonl_dataset.py     # jsonl 字节偏移索引 + 图片加载 + collate
├── models/
│   ├── backbone.py          # ResNet18/34/50，输出 /32 特征图
│   ├── common.py            # DualBranchHead（cls / box 双支路）+ 2D 正弦位置编码
│   └── detr_nar.py          # DETR 式非自回归模型
├── losses.py                # 交叉熵 + box L1 + GIoU（只在 mask=1 的位置算）
├── train.py                 # 训练入口
├── infer.py                 # 推理入口
├── tools/
│   ├── dashboard.py         # 训练监控前端（启停、曲线、日志、run 对比）
│   ├── verify_jsonl.py      # 标签转换正确性验证
│   ├── verify_otsl.py       # 标签体系全量验证
│   └── viz_otsl.py          # 原图 ↔ 重建表格 对照页
├── outputs/                 # 训练产物: <run>/{last.pt,best.pt,train_log.jsonl,config_snapshot.yaml}
└── logs/                    # 训练日志 <run>.log
```

---

## 4. 环境

```bash
python -m pip install torch pillow numpy pyyaml
python -m pip install pyarrow          # 只有 tools/verify_otsl.py 需要
```

训练在 WSL（Ubuntu-2204）里执行 `~/miniconda3/envs/llm/bin/python`；
Windows 侧 python 用于推理（CPU）和跑工具脚本。

---

## 5. 训练

### 5.1 命令行

```bash
# 从零训练
python train.py --batch-size 32 --workers 4 --pretrained --amp --epochs 20 --out run1

# 断点续训（接着 epoch+1 跑，优化器与 lr 调度一起恢复）
python train.py --resume outputs/run1/last.pt --epochs 20 --batch-size 32 --workers 4 --amp

# 小样本先跑通管线
python train.py --limit-train 500 --limit-val 40 --epochs 1
```

参数：`--epochs / --batch-size / --lr / --workers / --amp / --no-amp / --device / --out / --resume /
--limit-train / --limit-val`，其余全部来自 `config.yaml`，可临时覆盖：

```bash
python train.py --set data.img_size=896 model.dropout=0.2
```

关键配置（`config.yaml`）：

```yaml
data:   img_size 896 | max_len 500 | long_policy drop | batch_size 24 | num_workers 4
        pin_memory false | prefetch_factor 2
model:  backbone resnet34 | out_stride 32 | hidden_dim 512 | nheads 8
        shared_layers 2 | cls_layers 2 | box_branch_layers 3 | dim_feedforward 2048
        num_classes 8 | num_queries 500 | query_causal true | box_param cxywh
train:  epochs 20 | lr 3e-4 | lr_backbone 3e-5 | clip_max_norm 1.0 | warmup_steps 300
        lr_drop 14 | loss {cell 1.0, l1 2.0, giou 2.0}
```

`config.py` 启动时校验：`img_size` 能被 32 整除、`num_queries ≥ max_len`、
`num_classes == 标签表大小`、`hidden_dim % nheads == 0`、两个 jsonl 存在。

### 5.2 监控前端

```bash
python tools/dashboard.py --port 8126      # http://127.0.0.1:8126/
```

- 左侧填参数（epochs / batch / img / lr_drop / workers / 因果 mask 开关）→「开始训练」
- 右侧实时日志 + 每 epoch 曲线（loss / token_acc / seq_acc / span_acc / box_acc）
- 勾「续训」会带 `--resume outputs/<run>/last.pt`
- 下方 run 对比表：按 best token_acc 排序，标出每个 run 的 causal on/off

训练进程是控制台进程的子进程，**训练期间不要关控制台**。

### 5.3 指标

| 指标 | 含义 |
|---|---|
| `token_acc` | 逐 token 准确率（大量 `nl`/`fcel` 会把数值拉高，不能单独看） |
| `seq_acc` | **整张表结构全对的比例**，TSR 的真实准确率 |
| `span_acc` | GT 为 `<span>` 的位置预测对的比例 |
| `box_acc` | cell 框 IoU ≥ 0.5 的比例 |
| `box_iou` | 平均 IoU |

参考（50k 子集，896 输入，bs 32，第 15 epoch）：

```
val loss 1.245 | token_acc 93.34% | seq(整表) 56.50% | span_acc 51.15% | box_acc 68.15%
```

---

## 6. 推理

```bash
# 单张
python infer.py --image path/to/table.png

# 整个目录（默认取前 8 张）
python infer.py --dir path/to/images --limit 20

# 指定 checkpoint，额外输出 HTML，不画框图
python infer.py --image x.png --ckpt outputs/run1/best.pt --html --no-viz

# 用 jsonl 里的 filename 当清单
python infer.py --list path/to/val.jsonl --dir path/to/images --limit 50
```

默认 `--ckpt outputs/detr_otsl/best.pt`、`--out outputs/infer`，设备自动选（CPU 约 230 ms/张）。

每张图产出：

```
outputs/infer/<name>.json          # 结构化结果（主产物）
outputs/infer/<name>_boxes.png     # 原图 + 每个 cell 的框和编号
outputs/infer/<name>.html          # 由标签重建的表格（--html 时才有）
```

### 6.1 JSON 结构

```json
{
  "filename": "PMC1215490_002_00.png",
  "image_size": [503, 86],
  "input_size": 896,
  "scale": 0.8914,
  "rows": 6,
  "cols": 3,
  "n_cells": 18,
  "n_tokens": 24,
  "hit_eos": true,
  "mean_score": 0.9995,
  "tokens": ["ecel", "fcel", "fcel", "nl", "fcel", "..."],
  "html": "<table><tr><td></td><td></td><td></td></tr>...</table>",
  "cells": [
    {
      "id": 0, "r": 0, "c": 0,
      "rowspan": 1, "colspan": 1,
      "kind": "ecel",
      "token_index": 0,
      "bbox": [197.0, 4.5, 294.0, 14.1],
      "score": 0.9982
    }
  ],
  "warning": null
}
```

- `bbox` 已反算回**原图像素**（`x_norm × img_size / scale`），可直接裁图
- `cells` 只列**合并区左上角**（origin），用 `rowspan/colspan` 表达合并，不是每个网格位置都列
- `r/c` 网格坐标（0 起）；`bbox` 用于裁图与校对
- `kind`：`fcel` 有内容 / `ecel` 空 / `span` 合并区（`is_span` 或面积 > 1 都算）
- `warning` 记录「序列没出现 eos」「cell 标签数与网格 cell 数不一致」这类异常

### 6.2 代码里调用

```python
import torch
from config import get_config
from infer import load_model, infer_one

cfg = get_config()
model, cfg = load_model("outputs/detr_otsl/best.pt", cfg, torch.device("cpu"))
res = infer_one(model, cfg, torch.device("cpu"),
                "path/to/table.png", "outputs/infer", save_viz=True)
print(res["rows"], res["cols"], res["cells"][0]["bbox"])
```

### 6.3 转 docx

`cells` 的 `r / c / rowspan / colspan` 直接对应 python-docx：

```python
table = doc.add_table(rows=J["rows"], cols=J["cols"])
for c in J["cells"]:
    a = table.cell(c["r"], c["c"])
    if c["rowspan"] > 1 or c["colspan"] > 1:
        a = a.merge(table.cell(c["r"] + c["rowspan"] - 1, c["c"] + c["colspan"] - 1))
    a.text = c.get("text", "")        # 文字需另接 OCR
```

列宽按 `bbox` 的 x 边界聚类反推，不要等分。

---

## 7. 工具

| 命令 | 作用 |
|---|---|
| `python tools/verify_jsonl.py [--jsonl X --limit N]` | 逐条验证 `structure → 标签 → 网格 → HTML` 与原标注 HTML **逐字符相等** |
| `python tools/verify_otsl.py [--parquet-dir X]` | 标签体系全量验证（数据源为含官方 otsl 的 parquet） |
| `python tools/viz_otsl.py --jsonl X --n 6 --span-only` | 生成「原图 ↔ 由标签重建的空壳表格」对照页，人工复核 |
| `python tools/dashboard.py [--port 8126]` | 训练监控 + 启停 |

回归基线：

```
verify_jsonl    50000/50000 通过 (100%)，另 2 条参考 HTML 自身残缺，单列 official_defect
verify_otsl     394,944 条，纯净 99.9878%，48 条参考 HTML 行宽与 cols 冲突
```

---

## 8. 一致性约定

| 位置 | 约定 | 违反后果 |
|---|---|---|
| `models/*` 变量名 | 必须等于 checkpoint 的键名 | ckpt 无法加载 |
| `model.num_classes` | 等于 `datasets/otsl.py` 的 `NUM_CLASSES` | 分类头错位，`config.py` 启动即报错 |
| `model.num_queries` | ≥ `data.max_len` | 超长标签被截断，`config.py` 启动即报错 |
| `data.img_size` | 能被 32 整除 | backbone 特征图取整出错 |
| 数据集的 box 坐标系 | 固定为**输入图像素**（896 画布） | 训练侧再 ÷ img_size 归一化；改动需同步 `train.py` 与 `infer.py` |
| 标签表顺序 | `fcel/ecel/span/lcel/ucel/xcel/nl/eos` | 顺序变了等于重新定义类别，必须重训 |

**权重加载**直接用 PyTorch 原生的严格模式（`model.load_state_dict(ck, strict=True)`）：

- 键名不同、形状不同 → 直接抛错，不会加载一半
- `hidden_dim` / `num_classes` / `num_queries` / `backbone` / `out_stride` 变化都会改变权重的
  键名或形状，因此天然被拦住，不需要额外检查
- 只有 `data.img_size` 和 `model.box_param` 不体现在权重里，`train.py --resume` 会额外比一次：

```
续训中断: img_size 训练时是 896, 当前 config 是 768。
```

---

## 9. 已知限制

**a) 内存。** 当前机器 WSL 内存 15.9 GB。`bs 32 + 8 workers + pin_memory=true` 时数据加载
共享内存峰值约 9 GB，加上每个 epoch 末写两个 checkpoint（各 618 MB），容易在 epoch 边界被
OOM killer 杀掉（特征是日志无 traceback 突然中断）。默认配置为 `workers 4 + pin_memory false`
（整机峰值约 6.6 GB）。若需更大余量，在 `C:\Users\Administrator\.wslconfig` 设
`memory=24GB` 后执行 `wsl --shutdown`。

**b) 长表截断。** `num_queries = 500`，序列超过 500 的样本（约 0.18%）在 dataset 阶段被 drop
（`data.long_policy=drop`）。推理时未预测出 `<eos>` 会在 JSON 的 `warning` 里提示。

**c) `<span>` 不区分有无内容。** 有内容的合并起点与空的合并起点都标成 `span`，
需要区分时用别的信号判断。

**d) 少量样本的参考 HTML 与结构标注冲突。** 这类样本的 `structure` 完整、参考 HTML 行宽
与 `cols` 不符；转换以 `structure` 为准，验证脚本将其单列为 `official_defect`，不计失败。

**e) 空集保护。** `train.py` 在 `--limit-train` 小于一个 batch 且 `drop_last=true` 时会直接报错
（避免「跑完一个 epoch 但零步训练」）；`infer.py` 在目录为空时报错。
