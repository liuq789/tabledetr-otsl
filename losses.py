# -*- coding: utf-8 -*-
"""8 标签 OTSL 的损失: 结构 token 交叉熵 + box L1 + box GIoU.

pad id = 8, 而输出只有 8 类 (0..7) —— 8 超出类别范围, 不能当 ignore_index 用.
所以补齐位在 labels 里统一写 -100, loss 用 ignore_index=-100 排除.

序列形状
    OTSL 是定长网格: 长度 = rows*(cols+1), 没有 <SOS>/<EOS>.
    槽位 i 直接对第 i 个标签, 尾部空槽位是 -100 (不算 loss).
"""
import torch
import torch.nn.functional as F

IGNORE = -100


def token_ce(logits, targets, ignore_index=IGNORE):
    """logits (B,N,V), targets (B,N); targets==-100 的位置不计."""
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                           ignore_index=ignore_index)


def _pairwise_giou(pred, gt):
    """pred/gt: (M,4) xyxy 归一化 -> (M,) 的 GIoU."""
    px0, py0, px1, py1 = pred.unbind(-1)
    gx0, gy0, gx1, gy1 = gt.unbind(-1)

    area_p = (px1 - px0).clamp(min=0) * (py1 - py0).clamp(min=0)
    area_g = (gx1 - gx0).clamp(min=0) * (gy1 - gy0).clamp(min=0)

    iw = (torch.min(px1, gx1) - torch.max(px0, gx0)).clamp(min=0)
    ih = (torch.min(py1, gy1) - torch.max(py0, gy0)).clamp(min=0)
    inter = iw * ih
    union = area_p + area_g - inter + 1e-6
    iou = inter / union

    cw = torch.max(px1, gx1) - torch.min(px0, gx0)
    ch = torch.max(py1, gy1) - torch.min(py0, gy0)
    enclose = (cw * ch).clamp(min=1e-6)
    return iou - (enclose - union) / enclose


def pairwise_iou(pred, gt):
    """pred/gt: (M,4) xyxy 归一化 -> (M,) IoU."""
    px0, py0, px1, py1 = pred.unbind(-1)
    gx0, gy0, gx1, gy1 = gt.unbind(-1)
    area_p = (px1 - px0).clamp(min=0) * (py1 - py0).clamp(min=0)
    area_g = (gx1 - gx0).clamp(min=0) * (gy1 - gy0).clamp(min=0)
    iw = (torch.min(px1, gx1) - torch.max(px0, gx0)).clamp(min=0)
    ih = (torch.min(py1, gy1) - torch.max(py0, gy0)).clamp(min=0)
    inter = iw * ih
    return inter / (area_p + area_g - inter + 1e-6)


def box_losses(pred, gt, mask, use_giou=False):
    """pred/gt (B,N,4) xyxy 归一化, mask (B,N) 1=有框.

    与 TableMaster 一致: mask 之后做**分轴 L1** —— (x, w) 一组、(y, h) 一组.
    注意 OTSL 里只有 <fcel>/<ecel>/<span> 三类带框, lcel/ucel/xcel/nl 一律 mask=0.
    """
    m = mask.reshape(-1) > 0.5
    if m.sum() == 0:
        zero = pred.sum() * 0.0
        return zero, zero
    p, g = pred.reshape(-1, 4)[m], gt.reshape(-1, 4)[m]
    d = (p - g).abs()
    n = m.sum()
    l1 = (d[:, 0::2].sum() + d[:, 1::2].sum()) / n
    giou = (1.0 - _pairwise_giou(p, g)).mean() if use_giou else pred.sum() * 0.0
    return l1, giou


def box_metrics(pred, gt, mask, pred_tok=None, gt_tok=None, thr=0.5):
    """box 精度 (阈值 thr): iou_sum / n / acc_n / cell_n."""
    m = mask > 0.5
    n = int(m.sum())
    out = {"iou_sum": 0.0, "n": n, "acc_n": 0, "cell_n": 0}
    if n == 0:
        return out
    iou = pairwise_iou(pred[m], gt[m])
    out["iou_sum"] = float(iou.sum())
    out["acc_n"] = int((iou >= thr).sum())
    if pred_tok is not None:
        tok_ok = pred_tok[m].eq(gt_tok[m])
        out["cell_n"] = int((tok_ok & (iou >= thr)).sum())
    return out


def token_metrics(pred, tgt, ignore_index=IGNORE):
    """逐位 token 准确率 + 整条序列全对率 (尾部 -100 不参与).

    seq_acc 在 OTSL 里等价于"整张表的结构完全正确" —— 这就是 TSR 的主指标,
    比逐位准确率更能反映真实的表格还原能力.
    """
    keep = tgt.ne(ignore_index)
    tok_ok = int((pred.eq(tgt) & keep).sum())
    tok_all = int(keep.sum())
    seq_ok = int(((pred.eq(tgt) | ~keep).all(dim=1)).sum())
    return {
        "token_ok": tok_ok, "token_all": tok_all,
        "seq_ok": seq_ok, "seq_all": int(tgt.shape[0]),
    }


def tsr_loss(pred, targets, gt_boxes, box_mask, weights, aux_preds=None, aux_w=0.0,
             ignore_index=IGNORE):
    """统一损失入口 (结构 CE + 分轴 box L1 [+ 可选 GIoU]).

    pred      {"token_logits": (B,N,V), "boxes": (B,N,4), "aux": [...]}
    targets   (B,N) int64, -100 的位置不计
    gt_boxes  (B,N,4) 归一化坐标
    box_mask  (B,N) float, 1=该位有真实 cell 框
    weights   {"cell":1.0, "l1":5.0, "giou":0.0}   # key 兼容旧的 "ce"
    """
    ce_w = float(weights.get("cell", weights.get("ce", 1.0)))
    l1_w = float(weights.get("l1", weights.get("box_l1", 5.0)))
    gw = float(weights.get("giou", weights.get("box_giou", 0.0)))

    ce = token_ce(pred["token_logits"], targets, ignore_index)
    l1, giou = box_losses(pred["boxes"], gt_boxes, box_mask, use_giou=gw > 0)

    total = ce_w * ce + l1_w * l1 + gw * giou

    if aux_preds and aux_w > 0:
        for a in aux_preds:
            a_ce = token_ce(a[0], targets, ignore_index)
            a_l1, a_giou = box_losses(a[1], gt_boxes, box_mask, use_giou=gw > 0)
            total = total + aux_w * (ce_w * a_ce + l1_w * a_l1 + gw * a_giou)

    return total, {"ce": float(ce.detach()), "l1": float(l1.detach()),
                   "giou": float(giou.detach())}


__all__ = ["IGNORE", "token_ce", "pairwise_iou", "box_losses", "box_metrics",
           "token_metrics", "tsr_loss"]
