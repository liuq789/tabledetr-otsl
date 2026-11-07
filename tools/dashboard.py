# -*- coding: utf-8 -*-
"""训练控制台 (DETR 版) —— 模型结构 / 启动停止训练 / loss 与日志监控 / 断点续训.

单文件、只用标准库（不依赖 CDN，也不依赖 WSL 里的包），在 Windows 侧跑：

    python tools/dashboard.py --port 8126

三件事：
    1. 结构视图: /api/structure 返回示意图 + 分层参数表 + 真实模块树 (现场 build 一次模型)
    2. 训练控制: /api/start 起一个 wsl python train.py 子进程, /api/stop 结束它
    3. 监控: /api/state 汇总 outputs/*/train_log.jsonl 的 epoch 指标 + 日志尾巴

与 tabeldetr 版的差异
    * 只有 DETR 一个架构, 没有架构下拉框
    * 新增「因果 mask」开关: 勾上 -> model.query_causal=true (query 只看 0..i),
      取消 -> false (500 个 query 全互看, 尾部空槽位会参与注意力)
    * 指标列换成 OTSL 口径: token_acc / seq_acc(整表全对) / span_acc / box_acc
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
LOG_DIR = os.path.join(ROOT, "logs")
PY_WSL = os.environ.get("WB_PY_WSL", "~/miniconda3/envs/llm/bin/python")
WSL_DISTRO = os.environ.get("WB_WSL_DISTRO", "Ubuntu-2204")

_NAME_RE = re.compile(r"[^A-Za-z0-9_\-]")


def safe_name(n):
    """run 名只允许字母数字下划线连字符, 防止拼进 bash 命令造成注入."""
    return _NAME_RE.sub("", str(n or ""))


# 轮询缓存: /api/state 每 2s 被前端调一次, 若每次都真起 2~3 个 wsl 子进程,
# 页面会被拖死 (Windows 上单个 wsl 调用约 1~3s).
_STATE_CACHE = {"procs_t": 0.0, "procs_v": {}, "gpu_t": 0.0, "gpu_v": None}


def _reap_procs():
    """训练进程结束后关闭其日志文件句柄."""
    for name in list(PROCS):
        proc, f = PROCS[name]
        if proc.poll() is not None:
            try:
                f.close()
            except Exception:                                    # noqa: BLE001
                pass
            PROCS.pop(name, None)


def wsl_path(p):
    """D:\\work\\... / D:/work/... -> /mnt/d/work/..."""
    p = os.path.abspath(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


WSL_ROOT = wsl_path(ROOT)
PROCS_SH = WSL_ROOT + "/tools/_procs.sh"   # 进程识别脚本 (见文件内注释)

# ---------------------------------------------------------------- 结构示意

_STRUCT_CACHE = {}
_STRUCT_LOCK = threading.Lock()

_PROBE = """
import json, sys
sys.path.insert(0, '.')
from config import load_config
from models import build_model
cfg = load_config()
m = build_model(cfg)
rows = []
for name, child in m.named_children():
    n = sum(p.numel() for p in child.parameters())
    rows.append([name, type(child).__name__, int(n)])
total = sum(p.numel() for p in m.parameters())
print('<<<JSON>>>')
print(json.dumps({'rows': rows, 'total': int(total), 'text': repr(m)[:6000],
                  'query_causal': bool(cfg.model.get('query_causal', True)),
                  'img': int(cfg.data.img_size), 'queries': int(cfg.model.num_queries),
                  'classes': int(cfg.model.num_classes),
                  'shared': int(cfg.model.get('shared_layers', 2)),
                  'cls': int(cfg.model.get('cls_layers', 2)),
                  'box': int(cfg.model.get('box_branch_layers', 3))}))
"""


def probe_structure():
    """在 WSL 里现场 build 一次 DETR, 拿模块树 + 真实参数量 (带 mtime 缓存)."""
    key = (os.path.getmtime(os.path.join(ROOT, "config.yaml")),)
    with _STRUCT_LOCK:
        if key in _STRUCT_CACHE:
            return _STRUCT_CACHE[key]

    cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc",
           "cd %s && %s -W ignore -" % (WSL_ROOT, PY_WSL)]
    try:
        r = subprocess.run(cmd, input=_PROBE, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        out = r.stdout.split("<<<JSON>>>")
        data = json.loads(out[-1].strip()) if len(out) > 1 else \
            {"rows": [], "total": 0, "text": r.stderr[-2000:], "query_causal": True,
             "img": 0, "queries": 0, "trainable": 0}
    except Exception as exc:                                    # noqa: BLE001
        data = {"rows": [], "total": 0, "text": "build failed: %s" % exc,
                "query_causal": True, "img": 0, "queries": 0, "trainable": 0}

    qc = bool(data.get("query_causal", True))
    qn = data["queries"]
    blocks = [
        ("图像 %d x %d" % (data["img"], data["img"]), "输入", "in"),
        ("ResNet34 (/32)", "%d x %d = %d token -> K/V 记忆"
         % (data["img"] // 32, data["img"] // 32, (data["img"] // 32) ** 2), "bb"),
        ("+ 2D 正弦位置编码", "", "bb"),
        ("%d 个可学习 query" % qn, "输出槽位, 与输入 token 无关", "dec"),
        ("共享 Transformer x%d" % data["shared"], "query self-attn %s + cross-attn"
         % ("因果 (只看 0..i)" if qc else "全互看 (含尾部空槽位)"), "share"),
        ("cls 支路 x%d" % data["cls"], "Linear -> %d 类标签 (含 eos)" % data["classes"], "branch"),
        ("box 支路 x%d" % data["box"], "MLP -> 4 维 cell 框", "branch"),
        ("双分支输出", "%d x %d + %d x 4, 一次前向" % (qn, data["classes"], qn), "out"),
    ]
    data["blocks"] = [{"label": a, "sub": b, "kind": c} for a, b, c in blocks]
    with _STRUCT_LOCK:
        _STRUCT_CACHE[key] = data
    return data


# ---------------------------------------------------------------- 训练进程

PROCS = {}


def running_procs(force=False):
    """从 pgrep 里认领正在跑的训练, 返回 {run_name: 命令行}. 3s 缓存."""
    now = time.time()
    if not force and now - _STATE_CACHE["procs_t"] < 3:
        return _STATE_CACHE["procs_v"]
    # 认领正在跑的训练进程. 两个坑都要避开:
    #   1. pgrep 只给命令行 (python train.py --out xxx), 里面没有目录 -> 改用 /proc/<pid>/cwd
    #   2. wsl.exe 传参时会展开/吞掉内联脚本里的 $p, 导致 bash 语法错 (rc=2) ->
    #      脚本落成文件 tools/_procs.sh, 让 bash 从磁盘读
    cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", PROCS_SH]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    except Exception:                                            # noqa: BLE001
        return _STATE_CACHE["procs_v"]
    found, pids = {}, {}
    for line in r.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        pid, cwd, cmdline = parts[0].strip(), parts[1].strip(), parts[2].rstrip("\r")
        if not cwd.startswith(WSL_ROOT):                         # 只认本项目的工作目录
            continue
        m = re.search(r"--out\s+(\S+)", cmdline)
        if not m:
            continue
        name = m.group(1)
        found[name] = ("pid %s | %s" % (pid, cmdline.strip()))[:400]
        pids.setdefault(name, []).append(pid)
    _STATE_CACHE["procs_t"] = now
    _STATE_CACHE["procs_v"] = found
    _STATE_CACHE["pids_v"] = pids
    return found


def build_cmd(p):
    """拼 train.py 的命令行. 因果 mask 通过 --set model.query_causal 传."""
    causal = "true" if p.get("causal", True) else "false"
    inner = ("cd %s && PYTHONUNBUFFERED=1 %s -W ignore train.py "
             "--epochs %d --batch-size %d --workers %d --pretrained --amp "
             "--set data.img_size=%d train.lr_drop=%d train.log_every=100 "
             "model.query_causal=%s --out %s"
             % (WSL_ROOT, PY_WSL, int(p["epochs"]), int(p["batch"]),
                int(p.get("workers", 8)), int(p["img"]), int(p["drop"]),
                causal, p["name"]))
    if p.get("resume"):
        inner += " --resume outputs/%s/last.pt" % p["name"]
    return inner


def start_run(p):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "%s.log" % p["name"])
    mode = "a" if p.get("resume") else "w"
    if p.get("resume"):
        header = "\n\n===== resume @ %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        header = "===== start @ %s | query_causal=%s =====\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S"), p.get("causal", True))
    f = open(log_path, mode, encoding="utf-8", errors="replace")
    f.write(header)
    f.flush()
    proc = subprocess.Popen(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", build_cmd(p)],
        stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    PROCS[p["name"]] = (proc, f)
    return log_path


def stop_runs(name=None):
    name = safe_name(name)
    if name and name in PROCS:
        proc, f = PROCS.pop(name)
        try:
            proc.terminate()
        except Exception:                                        # noqa: BLE001
            pass
        try:
            f.close()
        except Exception:                                        # noqa: BLE001
            pass
    # 按 pid 停 (pgrep 命令行里没有目录, pkill 模式匹配不可靠): 主进程 + 全部 worker 一起停
    running_procs(force=True)
    pids_by_run = _STATE_CACHE.get("pids_v") or {}
    targets = []
    if name:
        targets = list(pids_by_run.get(name, []))
    else:
        for v in pids_by_run.values():
            targets.extend(v)
    if targets:
        subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc",
                        "kill -TERM %s 2>/dev/null; sleep 1; kill -KILL %s 2>/dev/null || true"
                        % (" ".join(targets), " ".join(targets))],
                       capture_output=True, timeout=30)
    _STATE_CACHE["procs_t"] = 0.0


# ---------------------------------------------------------------- 状态汇总

def _log_file_for(name):
    """一个 run 的日志可能来自两处:
    1. 控制台启动: logs/<name>.log (start_run 写的)
    2. 命令行启动: train_<name>.log 或 <name>.log (bash 重定向)
    按优先级找第一个存在的."""
    cands = [os.path.join(LOG_DIR, "%s.log" % name),
             os.path.join(ROOT, "train_%s.log" % name),
             os.path.join(ROOT, "%s.log" % name),
             os.path.join(OUT_DIR, name, "train.log")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def read_run(name):
    path = os.path.join(OUT_DIR, name, "train_log.jsonl")
    epochs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = r.get("val", {})
                epochs.append({
                    "epoch": r.get("epoch"),
                    "train_loss": r.get("train", {}).get("loss"),
                    "val_loss": v.get("loss"),
                    "token_acc": v.get("token_acc"),
                    "seq_acc": v.get("seq_acc"),
                    "span_acc": v.get("span_acc"),
                    "box_iou": v.get("box_iou"),
                    "box_acc": v.get("box_acc"),
                    "cell_acc": v.get("cell_acc"),
                    "ce": r.get("train", {}).get("ce"),
                    "sec": r.get("train_sec"),
                })
    tail = []
    log_file = _log_file_for(name)
    if log_file:
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                tail = [ln.rstrip("\n") for ln in f.readlines()[-120:]]
        except OSError:
            tail = []

    # 从配置快照里读因果 mask 开关, 便于对比不同设置下的 run
    qc = None
    snap = os.path.join(OUT_DIR, name, "config_snapshot.yaml")
    if os.path.exists(snap):
        with open(snap, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("query_causal:"):
                    qc = s.split(":", 1)[1].strip().lower() in ("true", "1", "yes")
                    break
    return {"name": name, "qc": qc, "epochs": epochs, "log": tail,
            "log_file": os.path.relpath(log_file, ROOT) if log_file else None,
            "best": max(epochs, key=lambda e: e["token_acc"] or 0) if epochs else None,
            "last": epochs[-1] if epochs else None,
            "has_ckpt": os.path.exists(os.path.join(OUT_DIR, name, "last.pt"))}


def list_runs():
    """列出所有可查看的 run: 有 train_log.jsonl 的历史 run + 正在跑但还没出第一个 epoch 的 run."""
    names = []
    if os.path.isdir(OUT_DIR):
        names = [d for d in os.listdir(OUT_DIR)
                 if os.path.exists(os.path.join(OUT_DIR, d, "train_log.jsonl"))]
    for n in running_procs():
        if n not in names and os.path.isdir(os.path.join(OUT_DIR, n)):
            names.append(n)
    runs = [read_run(n) for n in names]
    runs.sort(key=lambda r: max([e["epoch"] or 0 for e in r["epochs"]] or [0]), reverse=True)
    return runs


def gpu_info():
    now = time.time()
    if now - _STATE_CACHE["gpu_t"] < 5:
        return _STATE_CACHE["gpu_v"]
    try:
        r = subprocess.run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc",
                            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
                            "--format=csv,noheader,nounits | head -1"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        parts = [x.strip() for x in r.stdout.strip().split(",")]
        if len(parts) == 3:
            _STATE_CACHE["gpu_v"] = {"mem_used": int(parts[0]), "mem_total": int(parts[1]),
                                     "util": int(parts[2])}
            _STATE_CACHE["gpu_t"] = now
    except Exception:                                            # noqa: BLE001
        pass
    return _STATE_CACHE["gpu_v"]


# ---------------------------------------------------------------- HTTP

PAGE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OTSL 表格结构识别 · 训练控制台</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f6f7f9;color:#1f2329;font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;background:#fff;border-bottom:1px solid #e5e6eb}
h1{font-size:16px;font-weight:600;margin:0}
.pill{padding:2px 10px;border-radius:999px;font-size:12px;background:#eef1f6;color:#4e5969}
.pill.on{background:#e8f3ff;color:#1d6fe0}
.pill.off{background:#f2f3f5;color:#86909c}
.pill.qc{background:#eefaf3;color:#0f9d58}
.wrap{display:grid;grid-template-columns:320px 1fr;gap:14px;padding:14px 20px 40px}
.card{background:#fff;border:1px solid #e5e6eb;border-radius:10px;padding:14px}
.card h2{font-size:13px;font-weight:600;margin:0 0 10px;color:#4e5969}
label{display:block;font-size:12px;color:#4e5969;margin:8px 0 4px}
input,select{width:100%;padding:7px 9px;border:1px solid #dcdfe6;border-radius:7px;font-size:13px;background:#fff;color:#1f2329}
input[type=checkbox]{width:auto}
.row{display:flex;gap:10px}.row>*{flex:1}
button{border:0;border-radius:7px;padding:9px 12px;font-size:13px;font-weight:500;cursor:pointer}
#btnStart{background:#1d6fe0;color:#fff;width:100%;margin-top:12px}
#btnStop{background:#fbe9e9;color:#c0392b;width:100%;margin-top:8px}
button:disabled{opacity:.5;cursor:default}
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.node{border:1px solid #e5e6eb;border-radius:9px;padding:9px 12px;margin:0 auto;max-width:520px;background:#fafbfc}
.node .t{font-size:13px;font-weight:500}
.node .s{font-size:12px;color:#86909c}
.node.in{background:#eef4ff;border-color:#c9dcff}
.node.bb{background:#f3f7ff;border-color:#dbe6ff}
.node.enc{background:#efe9fe;border-color:#d5c9f8}
.node.share{background:#e8f6ef;border-color:#bfe6d4}
.node.dec{background:#fffaf0;border-color:#f5e2bb}
.node.branch{background:#fdeef4;border-color:#f6cade}
.node.out{background:#e8f3ff;border-color:#c2ddff}
.node.loop{background:#f2f3f5}
.arrow{height:16px;display:flex;align-items:center;justify-content:center;color:#c9cdd4;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:5px 6px;text-align:right;border-bottom:1px solid #f0f1f4;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:#86909c;font-weight:500}
canvas{width:100%;height:190px;display:block}
pre{margin:0;padding:10px;background:#1e1e1e;color:#d4d4d4;border-radius:8px;font:12px/1.55 Consolas,Menlo,monospace;max-height:260px;overflow:auto}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.empty{color:#86909c;font-size:12px;padding:18px 0;text-align:center}
.tablewrap{width:100%;overflow-x:auto}
.switch{display:flex;align-items:center;gap:8px;margin-top:10px;padding:8px 10px;background:#fafbfc;border:1px solid #e5e6eb;border-radius:7px}
.switch label{margin:0;font-size:12px}
.switch .hint{font-size:11px;color:#86909c;line-height:1.4}
</style></head><body>
<header>
  <h1>OTSL 表格结构识别 · 训练控制台</h1>
  <span class="pill off" id="pill">未在训练</span>
  <span class="pill" id="qcpill">causal -</span>
  <span class="pill" id="gpu">GPU -</span>
  <span class="pill" id="upd">-</span>
</header>
<div class="wrap">
  <div>
    <div class="card">
      <h2>训练控制</h2>
      <label>run 名称（输出目录 outputs/名称）</label>
      <input id="name" value="detr_otsl"/>
      <div class="row">
        <div><label>epochs</label><input id="epochs" type="number" value="20"/></div>
        <div><label>batch</label><input id="batch" type="number" value="24"/></div>
      </div>
      <div class="row">
        <div><label>img</label><input id="img" type="number" value="896"/></div>
        <div><label>lr_drop</label><input id="drop" type="number" value="14"/></div>
      </div>
      <div class="row">
        <div><label>workers</label><input id="workers" type="number" value="4"/></div>
        <div style="display:flex;align-items:flex-end;padding-bottom:8px">
          <label style="margin:0"><input type="checkbox" id="resume"/> 续训</label>
        </div>
      </div>
      <div class="switch">
        <input type="checkbox" id="causal" checked/>
        <label for="causal" style="flex:1">
          因果 mask（query 只看 0..i）
          <div class="hint" id="causalHint">关掉后 500 个 query 全互看：尾部空槽位也参与注意力</div>
        </label>
      </div>
      <button id="btnStart">开始训练</button>
      <button id="btnStop">停止训练</button>
      <div style="font-size:12px;color:#86909c;margin-top:10px" id="hint">续训需已有 outputs/&lt;run&gt;/last.pt</div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>模型结构</h2>
      <div id="struct"></div>
      <div style="margin-top:12px">
        <table id="params"></table>
      </div>
      <div style="margin-top:10px">
        <pre id="tree" style="max-height:200px">-</pre>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>查看已有 run</h2>
      <select id="runPick" style="max-width:340px"><option value="">— 选择 run —</option></select>
      <div style="font-size:12px;color:#86909c;margin-top:8px">
        选一个 run 才显示它的曲线/明细/日志；点左侧「开始训练」会自动切到新 run。
      </div>
    </div>

    <div id="resEmpty" class="card" style="margin-top:14px">
      <div class="empty" id="resEmptyMsg" style="padding:64px 0">
        还没有选择 run —— 先在上面选一个，或直接点「开始训练」
      </div>
    </div>

    <div id="resWrap" style="display:none">
    <div class="card" style="margin-top:14px">
      <h2>训练曲线 <span id="runName" style="font-weight:400;color:#86909c"></span></h2>
      <div class="grid2">
        <div><canvas id="lossChart"></canvas></div>
        <div><canvas id="accChart"></canvas></div>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>epoch 明细</h2>
      <div class="tablewrap"><table id="epTable"><thead><tr><th>epoch</th><th>train loss</th><th>val loss</th>
      <th>token_acc</th><th>seq_acc(整表)</th><th>span_acc</th><th>box_acc</th><th>box_IoU</th><th>秒</th></tr></thead><tbody></tbody></table></div>
      <div class="empty" id="epEmpty">还没有 epoch 记录</div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>日志（最后 120 行）<span id="logSrc" style="font-weight:400;color:#86909c;font-size:12px"></span></h2>
      <pre id="log">-</pre>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>run 对比（按历史最好 token_acc）</h2>
      <div class="tablewrap"><table id="cmpTable"><thead><tr><th>run</th><th>causal</th><th>ep</th>
      <th>best token_acc</th><th>best seq_acc</th><th>last cell_acc</th><th>last token_acc</th></tr></thead><tbody></tbody></table></div>
      <div class="empty" id="cmpEmpty">还没有训练记录</div>
    </div>
    </div>
  </div>
</div>
<script>
let state=null, activeRun=null;

function pct(x){ return x==null? '-' : (x*100).toFixed(2)+'%'; }
function num(x,d){ return x==null? '-' : Number(x).toFixed(d==null?3:d); }

async function api(path, body){
  const opt = body? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  return await r.json();
}

function drawChart(cv, series, opts){
  const dpr = window.devicePixelRatio||1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,w,h);
  const pad = {l:38,r:10,t:12,b:22};
  const W = w-pad.l-pad.r, H = h-pad.t-pad.b;
  const all = series.flatMap(s=>s.data.filter(v=>v!=null));
  if(!all.length){ g.fillStyle='#c9cdd4'; g.font='12px sans-serif'; g.textAlign='center';
    g.fillText('暂无数据', w/2, h/2); return; }
  let lo = Math.min(...all), hi = Math.max(...all);
  if(opts&&opts.pct){ lo=0; hi=Math.max(hi,lo+0.01); }
  if(hi-lo < 1e-9){ hi = lo + (opts&&opts.pct?0.1:1); }
  const m = (hi-lo)*0.08; hi += m; lo -= m;
  const n = Math.max(...series.map(s=>s.data.length), 2);
  const X = i => pad.l + (n<=1?0:W*i/(n-1));
  const Y = v => pad.t + H - H*(v-lo)/(hi-lo);

  g.strokeStyle='#f0f1f4'; g.lineWidth=1; g.font='11px sans-serif'; g.fillStyle='#a9aeb8';
  g.textAlign='right'; g.textBaseline='middle';
  for(let i=0;i<=4;i++){
    const y = pad.t + H*i/4, v = hi-(hi-lo)*i/4;
    g.beginPath(); g.moveTo(pad.l,y); g.lineTo(w-pad.r,y); g.stroke();
    g.fillText((opts&&opts.pct)? (v*100).toFixed(0)+'%' : v.toFixed(2), pad.l-6, y);
  }
  series.forEach(s=>{
    g.strokeStyle=s.color; g.lineWidth=1.8; g.beginPath();
    let started=false;
    s.data.forEach((v,i)=>{ if(v==null) return;
      if(!started){ g.moveTo(X(i),Y(v)); started=true; } else g.lineTo(X(i),Y(v)); });
    g.stroke();
    s.data.forEach((v,i)=>{ if(v==null) return;
      g.fillStyle=s.color; g.beginPath(); g.arc(X(i),Y(v),2.4,0,7); g.fill(); });
  });
  g.textAlign='left'; g.textBaseline='top'; let lx=pad.l;
  series.forEach(s=>{ g.fillStyle=s.color; g.fillRect(lx,2,8,8);
    g.fillStyle='#4e5969'; g.font='11px sans-serif'; g.fillText(s.name,lx+12,0); lx+=g.measureText(s.name).width+34; });
}

function renderStructure(){
  const box = document.getElementById('struct');
  box.innerHTML = '<div class="empty">构建中… (WSL 里现场 build 一次模型, 约 10~30s)</div>';
  document.getElementById('tree').textContent = '构建中…';
  api('/api/structure').then(d=>{
    box.innerHTML = d.blocks.map((b,i)=>
      (i? '<div class="arrow">&#9660;</div>':'') +
      '<div class="node '+b.kind+'"><div class="t">'+b.label+'</div>'+
      (b.sub?'<div class="s">'+b.sub+'</div>':'')+'</div>').join('');
    const tb = document.querySelector('#params');
    tb.innerHTML = '<tr><th>模块</th><th>类型</th><th>参数量</th></tr>' +
      d.rows.map(r=>'<tr><td>'+r[0]+'</td><td style="color:#86909c">'+r[1]+'</td><td>'+(r[2]/1e6).toFixed(2)+'M</td></tr>').join('') +
      '<tr><td><b>合计</b></td><td></td><td><b>'+(d.total/1e6).toFixed(2)+'M</b></td></tr>';
    document.getElementById('tree').textContent = d.text || '-';
  }).catch(e=>{ box.innerHTML='<div class="empty">读取失败</div>'; });
}

function render(){
  if(!state) return;
  const runs = state.runs||[];
  const running = state.running||{};
  const names = Object.keys(running);
  const pill = document.getElementById('pill');
  pill.className = 'pill ' + (names.length? 'on':'off');
  pill.textContent = names.length? ('训练中: '+names.join(', ')) : '未在训练';

  const qp = document.getElementById('qcpill');
  const qv = document.getElementById('causal').checked;
  qp.className = 'pill qc';
  qp.textContent = 'causal ' + (qv? 'on':'off');

  const g = state.gpu;
  document.getElementById('gpu').textContent = g? ('GPU '+g.mem_used+'/'+g.mem_total+' MB · '+g.util+'%') : 'GPU -';
  document.getElementById('upd').textContent = new Date().toLocaleTimeString();

  const pick = document.getElementById('runPick');
  const sig = runs.map(r=>r.name).join('|');
  if(pick.dataset.sig !== sig){
    pick.innerHTML = '<option value="">— 选择 run —</option>' +
      runs.map(r=>'<option value="'+r.name+'">'+r.name+
        (r.qc===null||r.qc===undefined?'':' · causal '+(r.qc?'on':'off'))+'</option>').join('');
    pick.dataset.sig = sig;
  }
  pick.value = activeRun || '';

  document.getElementById('resWrap').style.display = activeRun? 'block':'none';
  document.getElementById('resEmpty').style.display = activeRun? 'none':'block';
  const msg = document.getElementById('resEmptyMsg');
  if(!activeRun){ msg.textContent = '还没有选择 run —— 先在上面选一个，或直接点「开始训练」'; return; }

  const run = runs.find(r=>r.name===activeRun);
  if(!run){
    msg.textContent = names.indexOf(activeRun) >= 0
      ? ('正在启动 ' + activeRun + ' …（进程已起，等第一个 epoch 落盘）')
      : (activeRun + ' 既不在运行、也没有 train_log.jsonl —— 可能已结束或名字不对');
    document.getElementById('resWrap').style.display = 'none';
    document.getElementById('resEmpty').style.display = 'block';
    return;
  }
  document.getElementById('runName').textContent =
    '· '+run.name+(run.qc===null?'':' · causal '+(run.qc?'on':'off'));

  const eps = run.epochs || [];
  document.getElementById('epEmpty').style.display = eps.length? 'none':'block';
  document.querySelector('#epTable tbody').innerHTML = eps.map(e=>
    '<tr><td>'+e.epoch+'</td><td>'+num(e.train_loss)+'</td><td>'+num(e.val_loss)+'</td>'+
    '<td>'+pct(e.token_acc)+'</td><td><b>'+pct(e.seq_acc)+'</b></td><td>'+pct(e.span_acc)+'</td>'+
    '<td>'+pct(e.box_acc)+'</td><td>'+pct(e.box_iou)+'</td>'+
    '<td>'+(e.sec==null?'-':Math.round(e.sec))+'</td></tr>').join('');

  drawChart(document.getElementById('lossChart'), [
    {name:'train', color:'#1d6fe0', data: eps.map(e=>e.train_loss)},
    {name:'val', color:'#e8792b', data: eps.map(e=>e.val_loss)}
  ]);
  drawChart(document.getElementById('accChart'), [
    {name:'token', color:'#0f9d58', data: eps.map(e=>e.token_acc)},
    {name:'seq', color:'#8b5cf6', data: eps.map(e=>e.seq_acc)},
    {name:'span', color:'#d946a0', data: eps.map(e=>e.span_acc)},
    {name:'box', color:'#e05252', data: eps.map(e=>e.box_acc)}
  ], {pct:true});

  const log = document.getElementById('log');
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  if(run.log && run.log.length){
    log.textContent = run.log.join('\n');
    document.getElementById('logSrc').textContent = '· ' + run.log_file;
  } else {
    log.textContent = '日志还空着：刚启动的 run 一般要等几秒；如果一直是空，去 logs/ 或 train_<run>.log 看一眼';
    document.getElementById('logSrc').textContent = '';
  }
  if(atBottom) log.scrollTop = log.scrollHeight;

  const cmp = runs.slice().sort((a,b)=>((b.best&&b.best.token_acc)||0)-((a.best&&a.best.token_acc)||0));
  document.getElementById('cmpEmpty').style.display = cmp.length? 'none':'block';
  document.querySelector('#cmpTable tbody').innerHTML = cmp.map(r=>{
    const b = r.best||{}, l = r.last||{};
    return '<tr><td>'+r.name+'</td><td style="color:#86909c">'+
      (r.qc===null||r.qc===undefined?'-':(r.qc?'on':'off'))+'</td>'+
      '<td>'+(b.epoch||'-')+'</td><td><b>'+pct(b.token_acc)+'</b></td>'+
      '<td>'+pct(b.seq_acc)+'</td><td>'+pct(l.cell_acc)+'</td><td>'+pct(l.token_acc)+'</td></tr>';
  }).join('');
}

async function refresh(){ try{ state = await api('/api/state'); render(); }catch(e){} }

document.getElementById('runPick').addEventListener('change', e=>{
  activeRun = e.target.value || null;
  render();
});
document.getElementById('causal').addEventListener('change', e=>{
  document.getElementById('causalHint').textContent = e.target.checked
    ? 'query self-attention 因果：位置 i 只看 0..i，尾部空槽位不会污染真实位置'
    : '500 个 query 全互看：尾部空槽位也参与注意力，真实位置能"感知"序列长度';
  render();
});
document.getElementById('btnStart').addEventListener('click', async ()=>{
  const body = {
    name: document.getElementById('name').value.trim(),
    epochs: +document.getElementById('epochs').value,
    batch: +document.getElementById('batch').value,
    img: +document.getElementById('img').value,
    drop: +document.getElementById('drop').value,
    workers: +document.getElementById('workers').value,
    causal: document.getElementById('causal').checked,
    resume: document.getElementById('resume').checked
  };
  if(!body.name){ document.getElementById('hint').textContent = '失败: run 名不能为空'; return; }
  const r = await api('/api/start', body);
  document.getElementById('hint').textContent = r.ok? ('已启动 → logs/'+body.name+'.log') : ('失败: '+r.error);
  if(r.ok){
    activeRun = body.name;
    document.getElementById('runPick').dataset.sig = '';
    setTimeout(refresh, 500);
  }
});
document.getElementById('btnStop').addEventListener('click', async ()=>{
  const nm = activeRun || document.getElementById('name').value.trim() || null;
  if(!nm && !confirm('没选具体 run，将停止本项目所有训练进程。确认？')) return;
  const r = await api('/api/stop', {name: nm});
  document.getElementById('hint').textContent = r.ok? ('已发送停止信号'+(nm?(' → '+nm):' → 全部')) : ('失败: '+r.error);
  setTimeout(refresh, 800);
});

renderStructure();
refresh();
setInterval(refresh, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/state":
            _reap_procs()
            return self._send(200, {"runs": list_runs(), "running": running_procs(),
                                    "gpu": gpu_info()})
        if u.path == "/api/structure":
            return self._send(200, probe_structure())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if u.path == "/api/start":
            try:
                name = safe_name(body.get("name"))
                if not name:
                    return self._send(200, {"ok": False, "error": "run 名不能为空 (只允许字母数字 _ -)"})
                if running_procs().get(name):
                    return self._send(200, {"ok": False, "error": "该 run 已在训练中"})
                body = dict(body, name=name, causal=bool(body.get("causal", True)))
                if os.path.exists(os.path.join(OUT_DIR, name, "train_log.jsonl")) and not body.get("resume"):
                    return self._send(200, {"ok": False, "error": "同名 run 已存在，勾选续训或换名字"})
                if body.get("resume") and not os.path.exists(os.path.join(OUT_DIR, name, "last.pt")):
                    return self._send(200, {"ok": False, "error": "没有 last.pt 可续"})
                log = start_run(body)
                _STATE_CACHE["procs_t"] = 0.0
                return self._send(200, {"ok": True, "log": log})
            except Exception as exc:                             # noqa: BLE001
                return self._send(200, {"ok": False, "error": str(exc)})
        if u.path == "/api/stop":
            try:
                stop_runs(body.get("name"))
                return self._send(200, {"ok": True})
            except Exception as exc:                             # noqa: BLE001
                return self._send(200, {"ok": False, "error": str(exc)})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="训练控制台 (DETR / OTSL)")
    ap.add_argument("--port", type=int, default=8126)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("dashboard: http://%s:%d  (project %s)" % (a.host, a.port, ROOT))
    srv.serve_forever()


if __name__ == "__main__":
    main()
