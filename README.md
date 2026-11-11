# tabledetr-otsl

**English** | [中文](README.zh.md)

A table structure recognition (TSR) project. Given a table image, it predicts a **sequence of
structure labels plus a bbox for every cell**, and rebuilds the full grid (including
`rowspan` / `colspan`) from that label sequence.

The model decodes **non-autoregressively**, DETR style: a ResNet backbone produces a feature
sequence, and N learnable queries emit the entire label sequence in parallel — one forward pass
gives every result, with no teacher forcing and no recursive decoding.

```
image 896x896 -> ResNet34 (/32, 28x28 = 784 tokens) -> + 2D sinusoidal positional encoding
              -> 500 learnable queries + 2 shared layers -> split
                 ├─ cls branch, 2 layers -> 500 x 8  (7 grid labels + eos)
                 └─ box branch, 3 layers -> 500 x 4  (cell boxes, normalised 0..1)
```

51.50M parameters, outputs `(500, 8)` labels + `(500, 4)` boxes.

---

## 1. Label scheme (8 classes)

| id | label | meaning |
|---|---|---|
| 0 | `<fcel>` | cell with content |
| 1 | `<ecel>` | empty cell |
| 2 | `<span>` | top-left corner of a merged region (carries rowspan / colspan) |
| 3 | `<lcel>` | covered horizontally by the cell on the left |
| 4 | `<ucel>` | covered vertically by the cell above |
| 5 | `<xcel>` | covered from both left and above (inside a 2D merged region) |
| 6 | `<nl>` | end of row |
| 7 | `<eos>` | end of sequence |

- There is no thead / tbody and no parameter token such as `colspan="2"`. Rows = number of `<nl>`,
  columns = cells per row, and merge information is expressed implicitly by the distribution of
  `span` + `lcel/ucel/xcel`.
- Sequence length = `rows × (cols + 1) + 1`, strictly aligned with the grid, so it can be reshaped
  into `(rows, cols+1)` directly.
- `<eos>` lets the model learn where the sequence ends; inference stops on it, so row and column
  counts never have to be predicted separately.
- **Padding is not a class**: padded positions are written as `-100` in the labels and skipped by
  `cross_entropy(ignore_index=-100)`, so `num_classes = 8` covers every real class and costs no
  extra dimension.
- Labels contain **no text**. When text is needed, crop the image by the predicted bbox and run OCR.

Data-side statistics (50k subset): sequence length min 4 / median 60 / p95 231 / max 1300, and about
30% of tables contain merged cells.

---

## 2. Data format

`jsonl` annotations plus images of the same name in the same directory:

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

- `structure.tokens`: the table skeleton (`<td>` and `</td>` are separate tokens; tags with
  attributes are split into `<td` + ` colspan="2"` + `>`).
- `cells[i].bbox`: pixel coordinates in the original image. **Empty cells have no bbox field.**
- The i-th `<td` corresponds to `cells[i]`, one to one.

`datasets/convert.py` converts `structure` **on the fly** inside `__getitem__` (split by `<tr>` →
advance the grid according to rowspan → classify each position as fcel/ecel/span/lcel/ucel/xcel →
`nl` at the end of each row → `eos` at the very end), so no second annotation file is generated.
Data paths come from `config.yaml` (`paths.train_jsonl / val_jsonl`); when `paths.img_dir` is empty,
the directory of the jsonl is used.

---

## 3. Layout

```
tabledetr-otsl/
├── config.yaml              # every parameter (data / model / training), single source of truth
├── config.py                # config loading: get_config / parse_overrides / save_config / validate
├── datasets/
│   ├── otsl.py              # vocabulary, build_grid, grid_to_tokens, grid_to_html
│   ├── convert.py           # structure tokens -> label sequence
│   └── jsonl_dataset.py     # byte-offset index over the jsonl + image loading + collate
├── models/
│   ├── backbone.py          # ResNet18/34/50, /32 feature map
│   ├── common.py            # DualBranchHead (cls / box) + 2D sinusoidal positional encoding
│   └── detr_nar.py          # DETR-style non-autoregressive model
├── losses.py                # cross entropy + box L1 + GIoU (computed only where mask = 1)
├── train.py                 # training entry point
├── infer.py                 # inference entry point
├── tools/
│   ├── dashboard.py         # training dashboard (start/stop, curves, logs, run comparison)
│   ├── verify_jsonl.py      # verifies the label conversion
│   ├── verify_otsl.py       # full validation of the label scheme
│   └── viz_otsl.py          # side-by-side page: original image ↔ rebuilt table
├── outputs/                 # training artefacts: <run>/{last.pt,best.pt,train_log.jsonl,config_snapshot.yaml}
└── logs/                    # training logs <run>.log
```

---

## 4. Environment

```bash
python -m pip install torch pillow numpy pyyaml
python -m pip install pyarrow          # only needed by tools/verify_otsl.py
```

---

## 5. Training

### 5.1 Command line

```bash
# train from scratch
python train.py --batch-size 32 --workers 4 --pretrained --amp --epochs 20 --out run1

# resume (continues from epoch+1; optimiser and lr schedule are restored too)
python train.py --resume outputs/run1/last.pt --epochs 20 --batch-size 32 --workers 4 --amp

# smoke-test the pipeline on a small slice first
python train.py --limit-train 500 --limit-val 40 --epochs 1
```

Arguments: `--epochs / --batch-size / --lr / --workers / --amp / --no-amp / --device / --out /
--resume / --limit-train / --limit-val`. Everything else comes from `config.yaml` and can be
overridden ad hoc:

```bash
python train.py --set data.img_size=896 model.dropout=0.2
```

Key configuration (`config.yaml`):

```yaml
data:   img_size 896 | max_len 500 | long_policy drop | batch_size 24 | num_workers 4
        pin_memory false | prefetch_factor 2
model:  backbone resnet34 | out_stride 32 | hidden_dim 512 | nheads 8
        shared_layers 2 | cls_layers 2 | box_branch_layers 3 | dim_feedforward 2048
        num_classes 8 | num_queries 500 | query_causal true | box_param cxywh
train:  epochs 20 | lr 3e-4 | lr_backbone 3e-5 | clip_max_norm 1.0 | warmup_steps 300
        lr_drop 14 | loss {cell 1.0, l1 2.0, giou 2.0}
```

`config.py` validates at startup: `img_size` divisible by 32, `num_queries ≥ max_len`,
`num_classes == size of the label table`, `hidden_dim % nheads == 0`, and both jsonl files exist.

### 5.2 Dashboard

```bash
python tools/dashboard.py --port 8126      # http://127.0.0.1:8126/
```

- Fill in parameters on the left (epochs / batch / img / lr_drop / workers / causal-mask toggle) →
  press "start training".
- Live log on the right plus per-epoch curves (loss / token_acc / seq_acc / span_acc / box_acc).
- Tick "resume" to pass `--resume outputs/<run>/last.pt`.
- A run comparison table at the bottom, sorted by best token_acc, marking causal on/off per run.

The training process is a child of the console process — **do not close the console while training**.

### 5.3 Metrics

| metric | meaning |
|---|---|
| `token_acc` | per-token accuracy (lots of `nl`/`fcel` inflate it; never read it alone) |
| `seq_acc` | **fraction of tables whose structure is entirely correct** — the real TSR accuracy |
| `span_acc` | accuracy at positions where the ground truth is `<span>` |
| `box_acc` | fraction of cells with box IoU ≥ 0.5 |
| `box_iou` | mean IoU |

Reference run (50k subset, 896 input, bs 32, epoch 15):

```
val loss 1.245 | token_acc 93.34% | seq(full table) 56.50% | span_acc 51.15% | box_acc 68.15%
```

---

## 6. Inference

```bash
# single image
python infer.py --image path/to/table.png

# whole directory (first 8 images by default)
python infer.py --dir path/to/images --limit 20

# specific checkpoint, also emit HTML, skip the box visualisation
python infer.py --image x.png --ckpt outputs/run1/best.pt --html --no-viz

# use the filenames listed in a jsonl
python infer.py --list path/to/val.jsonl --dir path/to/images --limit 50
```

Defaults: `--ckpt outputs/detr_otsl/best.pt`, `--out outputs/infer`, device chosen automatically
(about 230 ms per image on CPU).

Per image it produces:

```
outputs/infer/<name>.json          # structured result (main artefact)
outputs/infer/<name>_boxes.png     # original image + a box and an index for every cell
outputs/infer/<name>.html          # table rebuilt from the labels (only with --html)
```

### 6.1 JSON structure

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

- `bbox` is mapped back to **original-image pixels** (`x_norm × img_size / scale`), ready for
  cropping.
- `cells` lists only the **top-left corner of each merged region** (the origin) and expresses merges
  through `rowspan/colspan`; it is not one entry per grid position.
- `r/c` are grid coordinates (0-based); `bbox` is for cropping and visual checks.
- `kind`: `fcel` has content / `ecel` empty / `span` merged region (`is_span` or area > 1).
- `warning` records anomalies such as "no eos in the sequence" or "cell label count does not match
  the number of grid cells".

### 6.2 Calling from code

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

### 6.3 Converting to docx

`r / c / rowspan / colspan` from `cells` map directly onto python-docx:

```python
table = doc.add_table(rows=J["rows"], cols=J["cols"])
for c in J["cells"]:
    a = table.cell(c["r"], c["c"])
    if c["rowspan"] > 1 or c["colspan"] > 1:
        a = a.merge(table.cell(c["r"] + c["rowspan"] - 1, c["c"] + c["colspan"] - 1))
    a.text = c.get("text", "")        # text needs a separate OCR pass
```

Derive column widths by clustering the x boundaries of `bbox` values; do not divide them evenly.

---

## 7. Tools

| command | purpose |
|---|---|
| `python tools/verify_jsonl.py [--jsonl X --limit N]` | validates `structure → labels → grid → HTML` against the original annotated HTML, character by character |
| `python tools/verify_otsl.py [--parquet-dir X]` | full validation of the label scheme (source: parquet files containing official OTSL) |
| `python tools/viz_otsl.py --jsonl X --n 6 --span-only` | builds a side-by-side page of "original image ↔ empty table rebuilt from labels" for manual review |
| `python tools/dashboard.py [--port 8126]` | training dashboard and process control |

## 8. Consistency rules

| where | rule | what breaks if violated |
|---|---|---|
| variable names in `models/*` | must match the checkpoint keys | the checkpoint cannot be loaded |
| `model.num_classes` | equals `NUM_CLASSES` in `datasets/otsl.py` | classification head misaligned; `config.py` fails at startup |
| `model.num_queries` | ≥ `data.max_len` | over-long label sequences get truncated; `config.py` fails at startup |
| `data.img_size` | divisible by 32 | backbone feature-map rounding breaks |
| dataset box coordinate system | always **input-image pixels** (the 896 canvas) | training divides by `img_size` afterwards; changing it requires updating both `train.py` and `infer.py` |
| label table order | `fcel/ecel/span/lcel/ucel/xcel/nl/eos` | a different order means redefining the classes — retrain from scratch |

**Weight loading** uses PyTorch's native strict mode (`model.load_state_dict(ck, strict=True)`):

- different key names or shapes → it raises immediately instead of loading half a model;
- changing `hidden_dim` / `num_classes` / `num_queries` / `backbone` / `out_stride` alters key names
  or shapes, so those mismatches are caught for free with no extra checks;
- only `data.img_size` and `model.box_param` are invisible to the weights, and `train.py --resume`
  performs an extra explicit comparison for those two.
