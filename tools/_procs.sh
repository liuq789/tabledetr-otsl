#!/bin/bash
# 列出所有 train.py 进程, 每行格式: pid|cwd|cmdline
# 为什么要放成文件: wsl.exe 传参时会展开/吞掉内联脚本里的 $ 变量,
# 导致 `for p in $(pgrep ...)` 这类写法直接语法错 (rc=2), 从文件读就没这个问题.
for p in $(pgrep -f 'train[.]py'); do
    cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null)
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    echo "$p|$cwd|$cmd"
done
