# -*- coding: utf-8 -*-
"""ResNet 视觉 backbone: 三个模型共用, 只负责"图像 -> 特征序列".

输出
    tokens (B, H*W, hidden_dim)   步长 32 的特征图展平
    (h, w)                        特征图空间尺寸, 用于生成二维位置编码

优先用 torchvision 的 resnet (可加载 ImageNet 预训练); 没有 torchvision 时
用本文件里的手写实现兜底 (结构完全一致, 只是拿不到预训练权重).
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 手写兜底

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * 4))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


class _ResNet(nn.Module):
    """ResNet18/34 (BasicBlock) / ResNet50 (Bottleneck), 去掉 fc, 输出 /32 特征图."""

    def __init__(self, name="resnet34"):
        super().__init__()
        cfg = {
            "resnet18": (_BasicBlock, [2, 2, 2, 2]),
            "resnet34": (_BasicBlock, [3, 4, 6, 3]),
            "resnet50": (_Bottleneck, [3, 4, 6, 3]),
        }[name]
        block, layers = cfg
        self.out_ch = 512 if block is _BasicBlock else 2048

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        chs = [64, 128, 256, 512]
        stages, in_ch = [], 64
        for i, (c, n) in enumerate(zip(chs, layers)):
            blocks = [block(in_ch, c, stride=1 if i == 0 else 2)]
            blocks += [block(c * block.expansion, c) for _ in range(n - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = c * block.expansion
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        x = self.stem(x)
        return self.stages(x)


# ---------------------------------------------------------------- 统一封装

# 各 stage 的输出通道: (resnet18/34, resnet50), idx: 1=layer1 .. 4=layer4
_STAGE_CH = {1: (64, 256), 2: (128, 512), 3: (256, 1024), 4: (512, 2048)}


class ResNetBackbone(nn.Module):
    """resnet -> 1x1 投影到 hidden_dim -> 展平成序列.

    out_stride: 32(默认) / 16 / 8, 决定用哪一级 stage 的特征图.
        32 -> layer4, 16 -> layer3, 8 -> layer2.
        表格细结构多, /32 一个 token 盖 32x32 像素, 小单元格会被糊掉;
        需要更细的特征时可试 /16 (token 数 x4, cross-attention 开销随 n 线性增长).
    """

    def __init__(self, name="resnet34", hidden_dim=256, pretrained=False,
                 dropout=0.0, out_stride=32):
        super().__init__()
        stages, final_ch = _build_stages(name, pretrained)
        self.stages = nn.ModuleList(stages)          # [stem(/4), layer1..layer4]
        n_use = {32: 5, 16: 4, 8: 3}.get(int(out_stride))
        if n_use is None:
            raise ValueError("out_stride 只支持 32/16/8, 当前 %r" % out_stride)
        self.n_use = n_use
        self.out_ch = _STAGE_CH[n_use - 1][0 if name in ("resnet18", "resnet34") else 1]

        self.proj = nn.Sequential(
            nn.Conv2d(self.out_ch, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, img):
        """img (B,3,H,W) -> tokens (B,HW,C), (h,w)."""
        x = img
        for st in self.stages[:self.n_use]:
            x = st(x)
        f = self.proj(x)
        B, C, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2)
        return self.drop(tokens), (h, w)


def _build_stages(name, pretrained):
    """统一返回 [stem, layer1..layer4] 列表与最终输出通道."""
    stages, out_ch = _build_torchvision_stages(name, pretrained)
    if stages is None:
        net = _ResNet(name)
        out_ch = net.out_ch
        stages = [net.stem] + [s for s in net.stages.children()]
    return stages, out_ch


def _build_torchvision_stages(name, pretrained):
    try:
        import torchvision.models as tvm
    except ImportError:
        return None, None

    fn = {"resnet18": tvm.resnet18, "resnet34": tvm.resnet34,
          "resnet50": tvm.resnet50}.get(name)
    if fn is None:
        raise ValueError("backbone 只支持 resnet18/34/50, 当前 %r" % name)

    weights = None
    if pretrained:
        w = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights",
             "resnet50": "ResNet50_Weights"}[name]
        weights = getattr(tvm, w).DEFAULT
    net = fn(weights=weights)
    out_ch = net.fc.in_features
    # 注意: 不能只把 fc 换成 Identity —— torchvision 的 forward 里还有 avgpool+flatten,
    # 输出会变成 (B,out_ch) 向量. 这里把 stem + 4 个 stage 摘出来自己串, 才能拿到特征图.
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    return [stem, net.layer1, net.layer2, net.layer3, net.layer4], out_ch


def build_backbone(cfg):
    m = cfg.model
    return ResNetBackbone(name=m.backbone, hidden_dim=int(m.hidden_dim),
                          pretrained=bool(m.get("pretrained", False)),
                          dropout=float(m.get("dropout", 0.0)),
                          out_stride=int(m.get("out_stride", 32)))
