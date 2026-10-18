# -*- coding: utf-8 -*-
"""从 skypile_00000.parquet 抽取高中文占比的网页文本 ~150MB -> cpp/skypile_cn.txt"""
import pyarrow.parquet as pq
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET = 150 * 1000 * 1000

t = pq.read_table(os.path.join(ROOT, "skypile_00000.parquet"), columns=["text"])
texts = t.column("text").to_pylist()
print("总行数:", len(texts))

def cn_ratio(s):
    n = len(s)
    if n == 0:
        return 0.0
    cn = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
    return cn / n

out, total, kept = [], 0, 0
for s in texts:
    if total >= TARGET:
        break
    n = len(s)
    if n < 200 or n > 20000:      # 太短没营养，太长可能是网页堆叠
        continue
    if cn_ratio(s) < 0.6:         # 【记】只要以中文为主体的网页
        continue
    out.append(s)
    total += n
    kept += 1

blob = "\n\n".join(out).encode("utf-8")
path = os.path.join(ROOT, "cpp", "skypile_cn.txt")
open(path, "wb").write(blob)
print(f"保留 {kept} 行，共 {len(blob)/1e6:.1f}MB -> {path}")
print("抽样:", blob[3000:3120].decode("utf-8", errors="replace"))
