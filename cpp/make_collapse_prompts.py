# -*- coding: utf-8 -*-
"""坍缩实验：生成纯换行提示文件（2048 字节 0x0A），零真实数据泄漏。"""
import sys

out = "prompts_nl.bin"
n = 2048
with open(out, "wb") as f:
    f.write(b"\x0a" * n)
print("written", out, n, "bytes")
