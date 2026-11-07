# -*- coding: utf-8 -*-
"""DETR 式 OTSL(8 标签, 含 eos) 表格结构识别训练入口.

    python train.py                                  # 按 config.yaml 跑
    python train.py --batch-size 32 --amp            # 常用覆盖
    python train.py --limit-train 64 --epochs 1      # 先小样本跑通管线
    python train.py --resume outputs/detr/last.pt

模型: ResNet backbone + N 个可学习 query, 共享层后分叉成 cls / box 双支路,
一次前向并行出全部槽位 (非自回归, 无匈牙利匹配).

与 tabeldetr 的差异只来自"7 个网格标签 + 定长网格"这个设定:
    * 标签序列 = rows*(cols+1) 个网格 token + 1 个 <eos> (没有 <SOS>)
      eos 让模型学会"序列到哪结束", 推理时遇到 eos 即停, 不需要额外预测行列数
    * pad id = 7 超出 7 类输出范围, 所以补齐位写 -100, loss 用 ignore_index=-100
    * 数据集是 jsonl + 同名 png (与 tabeldetr 格式一致), structure 现场转 7 个网格标签

产物 (outputs/detr/): last.pt / best.pt / train_log.jsonl / config_snapshot.yaml
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from config import load_config, parse_overrides, save_config, summary, to_plain
from datasets.otsl import IGNORE_INDEX, NUM_CLASSES, SPAN, STOI
from datasets.jsonl_dataset import build_dataloader as build_otsl_loader
from losses import box_metrics, token_metrics, tsr_loss
from models import build_model, count_params

# ---------------------------------------------------------------- 工具

ARCH = "detr"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(arg):
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tune_cuda(device):
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def make_loader(split, cfg, limit=None):
    """训练/验证 DataLoader (map-style, shuffle 用 RandomSampler)."""
    dl = build_otsl_loader(split, cfg=cfg, with_bbox=True)
    if not limit:
        return dl
    idx = list(range(min(int(limit), len(dl.dataset))))
    print("  [%s] 限制 %d 条" % (split, len(idx)))
    return DataLoader(Subset(dl.dataset, idx), batch_size=dl.batch_size,
                      shuffle=(split == "train"), drop_last=(split == "train"),
                      num_workers=dl.num_workers, collate_fn=dl.collate_fn,
                      pin_memory=dl.pin_memory, persistent_workers=False)


def normalize_boxes(boxes, img_size):
    """input 像素坐标 -> 0~1, 让 L1/GIoU 的尺度稳定."""
    return (boxes / float(img_size)).clamp(0.0, 1.0)


# ---------------------------------------------------------------- 一步前向

def forward_batch(model, batch, device, img_size):
    """返回 (pred, targets, gt_boxes, box_mask); targets 里的 -100 是补齐位.

    模型输出固定 num_queries 个槽位, 这里把标签对齐到槽位数: 超长截断, 不足补 -100.
    """
    img = batch["img"].to(device, non_blocking=True)
    tgt = batch["labels"].to(device)
    boxes = normalize_boxes(batch["boxes"].to(device), img_size)
    bmask = batch["box_mask"].to(device)

    out = model(img)
    n_slot = out["token_logits"].shape[1]
    n_tok = tgt.shape[1]
    if n_tok > n_slot:
        tgt = tgt[:, :n_slot]
        boxes, bmask = boxes[:, :n_slot], bmask[:, :n_slot]
    elif n_tok < n_slot:
        tgt = torch.cat([tgt, tgt.new_full((tgt.shape[0], n_slot - n_tok), IGNORE_INDEX)], 1)
        boxes = torch.cat([boxes, boxes.new_zeros((boxes.shape[0], n_slot - n_tok, 4))], 1)
        bmask = torch.cat([bmask, bmask.new_zeros((bmask.shape[0], n_slot - n_tok))], 1)
    return out, tgt, boxes, bmask


# ---------------------------------------------------------------- 评估

@torch.no_grad()
def evaluate(model, dl, cfg, device):
    """返回 loss / token_acc / seq_acc(整表全对) / span_acc / box 指标."""
    model.eval()
    w = cfg.train.loss
    weights = {"cell": float(w.get("cell", 1.0)),
               "l1": float(w.get("l1", 5.0)),
               "giou": float(w.get("giou", 0.0))}
    img_size = int(cfg.data.img_size)
    span_id = STOI[SPAN]

    losses = []
    tok_ok = tok_all = seq_ok = seq_all = 0
    span_ok = span_all = 0
    iou_sum = iou_n = box_ok = cell_ok_n = 0

    for batch in dl:
        out, tgt, gt_box, gmask = forward_batch(model, batch, device, img_size)
        loss, _ = tsr_loss(out, tgt, gt_box, gmask, weights, ignore_index=IGNORE_INDEX)
        losses.append(float(loss))

        pred = out["token_logits"].argmax(-1)
        m = token_metrics(pred, tgt, ignore_index=IGNORE_INDEX)
        tok_ok += m["token_ok"]; tok_all += m["token_all"]
        seq_ok += m["seq_ok"]; seq_all += m["seq_all"]

        is_span = tgt.eq(span_id)                 # -100 不等于 span, 天然排除补齐位
        span_all += int(is_span.sum())
        span_ok += int((pred.eq(span_id) & is_span).sum())

        bm = box_metrics(out["boxes"], gt_box, gmask, pred, tgt)
        iou_sum += bm["iou_sum"]; iou_n += bm["n"]
        box_ok += bm["acc_n"]; cell_ok_n += bm["cell_n"]

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "token_acc": tok_ok / max(tok_all, 1),
        "seq_acc": seq_ok / max(seq_all, 1),
        "span_acc": (span_ok / span_all) if span_all else None,
        "box_iou": (iou_sum / iou_n) if iou_n else None,
        "box_acc": (box_ok / iou_n) if iou_n else None,
        "cell_acc": (cell_ok_n / iou_n) if iou_n else None,
        "n_batch": len(losses),
    }


# ---------------------------------------------------------------- 训练

def train(args):
    cfg = load_config(args.config, parse_overrides(args.overrides))

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.pretrained:
        cfg["model"]["pretrained"] = True
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp

    set_seed(int(cfg.project.seed))
    device = pick_device(args.device)
    tune_cuda(device)
    out_dir = os.path.join(cfg.project.output_dir, args.out or ARCH)

    print("=" * 78)
    print("model   : DETR-NAR (ResNet + %d query, 并行解码)" % int(cfg.model.num_queries))
    print("labels  : %d 类 (7 网格标签 + <eos>), pad 不在其中 (补齐位 = -100)" % NUM_CLASSES)
    print("config  : %s" % summary(cfg))
    print("device  : %s | amp=%s" % (device, bool(cfg.train.amp) and device.type == "cuda"))
    print("out_dir : %s" % out_dir)
    print("=" * 78)

    train_dl = make_loader("train", cfg, args.limit_train)
    val_dl = make_loader("val", cfg, args.limit_val)
    print("train batches %d | val batches %d" % (len(train_dl), len(val_dl)))
    # 空 loader 会"跑完一个 epoch 但一步没训", 指标还照样写日志, 必须拦住
    if len(train_dl) == 0:
        raise SystemExit(
            "训练集切出 0 个 batch: --limit-train %s 不足一个 batch_size=%d,"
            " 而 data.drop_last=true 会把零头丢掉.\n"
            "  解决: 调大 --limit-train / 调小 --batch-size / 把 config 里 data.drop_last 设为 false"
            % (args.limit_train, train_dl.batch_size))
    if len(val_dl) == 0:
        raise SystemExit("验证集切出 0 个 batch: 检查 --limit-val 与 data.eval_batch_size")

    model = build_model(cfg).to(device)
    total, trainable = count_params(model)
    print("params: %.2fM (trainable %.2fM)" % (total / 1e6, trainable / 1e6))
    os.makedirs(out_dir, exist_ok=True)          # loader/model 都过了再建目录

    bb = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in bb_ids],
         "lr": float(cfg.train.lr)},
        {"params": bb, "lr": float(cfg.train.lr_backbone)},
    ]
    opt = AdamW(groups, weight_decay=float(cfg.train.weight_decay))

    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    lr_drop = int(cfg.train.lr_drop)
    steps_per_epoch = max(len(train_dl), 1)

    def lr_lambda(step):
        w = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        return w * (0.1 ** int(step // steps_per_epoch >= lr_drop))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, -1.0
    resume = args.resume or cfg.paths.get("resume")
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])      # 严格加载: 键名或形状不符直接抛错
        _check_non_weight_cfg(ck.get("cfg") or {}, cfg)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best", -1.0)
        print("resume from %s (epoch %d, best %.4f)" % (resume, start_epoch, best))

    save_config(cfg, os.path.join(out_dir, "config_snapshot.yaml"))
    log_path = os.path.join(out_dir, "train_log.jsonl")
    w = cfg.train.loss
    weights = {"cell": float(w.get("cell", 1.0)),
               "l1": float(w.get("l1", 5.0)),
               "giou": float(w.get("giou", 0.0))}
    aux_w = float(w.get("aux", 0.0))
    img_size = int(cfg.data.img_size)
    log_every = int(cfg.train.log_every)
    epochs = int(cfg.train.epochs)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        run = {"loss": 0.0, "ce": 0.0, "l1": 0.0, "giou": 0.0}
        n_step = 0

        for step, batch in enumerate(train_dl, 1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out, tgt, gt_box, gmask = forward_batch(model, batch, device, img_size)
                loss, comps = tsr_loss(out, tgt, gt_box, gmask, weights,
                                       aux_preds=out.get("aux"), aux_w=aux_w,
                                       ignore_index=IGNORE_INDEX)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip_max_norm))
            scaler.step(opt)
            scaler.update()
            sched.step()

            n_step += 1
            run["loss"] += float(loss.detach())
            for k in ("ce", "l1", "giou"):
                run[k] += comps[k]

            if step % log_every == 0:
                lr = opt.param_groups[0]["lr"]
                print("  ep %02d | %4d/%4d | loss %.3f (ce %.3f l1 %.3f giou %.3f) "
                      "| lr %.1e | %.1f it/s"
                      % (epoch + 1, step, len(train_dl),
                         run["loss"] / n_step, run["ce"] / n_step,
                         run["l1"] / n_step, run["giou"] / n_step,
                         lr, n_step / max(time.time() - t0, 1e-6)))

        train_time = time.time() - t0
        m = {k: v / max(n_step, 1) for k, v in run.items()}

        met = evaluate(model, val_dl, cfg, device)
        score = met["token_acc"]
        best = max(best, score)

        rec = {"epoch": epoch + 1, "train": m, "train_sec": round(train_time, 1),
               "val": met, "best": best}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print("ep %02d done in %.0fs | train loss %.3f | val loss %.3f | "
              "token_acc %.2f%% | seq(整表) %.2f%% | span_acc %.2f%% | box_acc %.2f%%"
              % (epoch + 1, train_time, m["loss"], met["loss"],
                 met["token_acc"] * 100, met["seq_acc"] * 100,
                 (met["span_acc"] or 0) * 100, (met["box_acc"] or 0) * 100))

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "best": best,
              "cfg": to_plain(cfg)}
        torch.save(ck, os.path.join(out_dir, "last.pt"))
        if score >= best - 1e-12:
            torch.save(ck, os.path.join(out_dir, "best.pt"))
            print("  * best updated: token_acc %.2f%% -> %s"
                  % (score * 100, os.path.join(out_dir, "best.pt")))

    print("training finished. logs: %s" % log_path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DETR 式 OTSL 表格结构识别训练 (7 个网格标签 + eos)")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="覆盖配置, 如 model.backbone=resnet50 data.img_size=896")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--pretrained", action="store_true", help="加载 ImageNet ResNet 权重")
    ap.add_argument("--amp", dest="amp", action="store_true", default=None)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="outputs/ 下的子目录名")
    ap.add_argument("--resume", default=None, help="checkpoint 路径")
    return ap.parse_args(argv)


def _check_non_weight_cfg(ck_cfg, cfg):
    """img_size / box_param 不体现在权重形状里, load_state_dict 抓不到, 单独比一次.

    其余结构参数 (hidden_dim / num_classes / num_queries / backbone / out_stride)
    变了必然导致权重形状不符, load_state_dict 会直接报错, 不需要额外检查.
    """
    for sec, key in (("data", "img_size"), ("model", "box_param")):
        old = (ck_cfg.get(sec) or {}).get(key)
        now = (cfg.get(sec) or {}).get(key)
        if old is not None and now is not None and old != now:
            raise SystemExit("续训中断: %s 训练时是 %s, 当前 config 是 %s。\n"
                             "  这两项不改变权重形状, 但会让结果错; 改回训练时的值, 或从零训练。"
                             % (key, old, now))


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
