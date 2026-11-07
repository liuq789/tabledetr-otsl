# -*- coding: utf-8 -*-
"""配置加载: 全项目参数唯一来源 config.yaml.

    from config import get_config
    cfg = get_config()                 # 单例, 首次读 config.yaml
    cfg.data.img_size                  # 属性访问
    cfg["model"]["hidden_dim"]         # 字典访问

命令行 / 脚本覆盖:
    cfg = get_config(overrides={"data.img_size": 896})
    cfg = build_config(["-c", "config.yaml", "--set", "train.epochs=10"])

相对路径 (paths 段) 一律相对 config.yaml 所在目录解析成绝对路径.
"""
import argparse
import copy
import os
import sys

try:
    import yaml
except ImportError:                                    # pragma: no cover
    raise ImportError("需要 pyyaml: python -m pip install pyyaml")

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "config.yaml")

_CACHE = None


# ---------------------------------------------------------------- 容器

class AttrDict(dict):
    """支持 cfg.a.b 与 cfg['a']['b'] 两种访问."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("配置里没有字段: %s" % key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    def dump(self):
        return to_plain(self)


def _wrap(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


# ---------------------------------------------------------------- 覆盖

def _set_by_path(cfg, dotted, value):
    """cfg['a.b.c'] = value (中间层不存在则创建)."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = AttrDict()
        node = node[k]
    node[keys[-1]] = value


def parse_overrides(items):
    """['data.img_size=896', 'model.pretrained=true'] -> {'data.img_size': 896, ...}"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError("覆盖参数要写成 key=value: %r" % it)
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v.strip())
    return out


# ---------------------------------------------------------------- 解析路径

def _resolve_paths(cfg, base_dir):
    """paths 段里的相对路径 -> 绝对路径 (null 保持 null)."""
    paths = cfg.get("paths") or AttrDict()
    for k, v in list(paths.items()):
        if isinstance(v, str) and v and not os.path.isabs(v):
            paths[k] = os.path.normpath(os.path.join(base_dir, v))
    out_dir = (cfg.get("project") or {}).get("output_dir")
    if isinstance(out_dir, str) and out_dir and not os.path.isabs(out_dir):
        cfg["project"]["output_dir"] = os.path.normpath(os.path.join(base_dir, out_dir))
    return cfg


# ---------------------------------------------------------------- 校验

def validate(cfg):
    """配置自洽性检查, 早失败好过训练到一半炸."""
    errs = []
    d, m, t = cfg.data, cfg.model, cfg.train

    if int(d.img_size) % 32 != 0:
        errs.append("data.img_size=%s 必须能被 32 整除 (backbone /32 步长)" % d.img_size)
    if int(d.max_len) < 2:
        errs.append("data.max_len=%s 太小" % d.max_len)
    if d.long_policy not in ("drop", "truncate"):
        errs.append("data.long_policy 只能是 drop / truncate, 当前 %r" % d.long_policy)
    if int(m.hidden_dim) % int(m.nheads) != 0:
        errs.append("model.hidden_dim=%s 不能被 nheads=%s 整除" % (m.hidden_dim, m.nheads))

    # decoder 的输出槽位必须放得下最长序列, 否则训练侧会静默截断标签
    if int(m.num_queries) < int(d.max_len):
        errs.append("model.num_queries=%s 小于 data.max_len=%s, 超出的标签会被丢掉"
                    % (m.num_queries, d.max_len))

    # num_classes 必须等于标签表的真实类别数 (不写死数字, 直接问词表)
    try:
        from datasets.otsl import NUM_CLASSES as _N_CLS
        if int(m.num_classes) != _N_CLS:
            errs.append("model.num_classes=%s 与标签表 datasets/otsl.py 的 NUM_CLASSES=%d 不一致"
                        % (m.num_classes, _N_CLS))
    except Exception:                                        # noqa: BLE001
        pass

    for p in ("train_jsonl", "val_jsonl"):
        v = cfg.paths.get(p)
        if not v or not os.path.exists(v):
            errs.append("paths.%s 不存在: %s" % (p, v))

    if errs:
        raise ValueError("config.yaml 校验失败:\n  - " + "\n  - ".join(errs))
    return cfg


# ---------------------------------------------------------------- 加载

def load_config(path=None, overrides=None, strict=True):
    """读 yaml -> 包装 -> 路径解析 -> (校验) -> 应用覆盖."""
    path = os.path.abspath(path or DEFAULT_CONFIG)
    if not os.path.exists(path):
        raise FileNotFoundError("找不到配置文件: %s" % path)

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _wrap(copy.deepcopy(raw))
    cfg["_config_path"] = path
    _resolve_paths(cfg, os.path.dirname(path))

    if overrides:
        for k, v in overrides.items():
            _set_by_path(cfg, k, v)

    if strict:
        validate(cfg)

    return cfg


def get_config(path=None, overrides=None, reload=False):
    """进程内单例: 各处调用拿到同一份配置."""
    global _CACHE
    if _CACHE is None or reload or path is not None or overrides:
        _CACHE = load_config(path, overrides)
    return _CACHE


def save_config(cfg, path):
    """把实际生效的配置另存一份 (训练目录留档用)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, allow_unicode=True, sort_keys=False)
    return path


def build_config(argv=None):
    """脚本统一入口: python xxx.py -c other.yaml --set data.img_size=896"""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args, _ = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    return get_config(args.config, parse_overrides(args.overrides), reload=True)


# ---------------------------------------------------------------- 打印

def summary(cfg):
    """一行式关键参数, 训练开始时打日志用."""
    m = cfg.model
    return ("img %d | max_len %d | classes %d | backbone %s | hidden %d | "
            "layers shared/cls/box %d/%d/%d | queries %d | bs %d | lr %.1e"
            % (cfg.data.img_size, cfg.data.max_len, m.num_classes,
               m.backbone, m.hidden_dim,
               m.get("shared_layers", 2), m.get("cls_layers", 2),
               m.get("box_branch_layers", 3), m.num_queries,
               cfg.data.batch_size, cfg.train.lr))


if __name__ == "__main__":
    c = build_config()
    print("config:", c["_config_path"])
    for sec in ("project", "paths", "data", "model", "train"):
        print("\n[%s]" % sec)
        for k, v in to_plain(c[sec]).items():
            print("  %-22s %s" % (k, v))
    print("\n" + summary(c))
