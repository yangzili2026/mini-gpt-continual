# -*- coding: utf-8 -*-
"""把 zhwiki_dataset.jsonl 提取成纯文本语料 -> cpp/zhwiki_cn.txt
过滤：正文 200~20000 字符的条目（去掉消歧义页/超短 stub），封顶 300MB。"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET = 300 * 1000 * 1000
out_path = os.path.join(ROOT, "cpp", "zhwiki_cn.txt")

n, kept, total = 0, 0, 0
out = []
with open(os.path.join(ROOT, "zhwiki_dataset.jsonl"), encoding="utf-8") as f:
    for line in f:
        n += 1
        try:
            r = json.loads(line)
        except Exception:
            continue
        c = (r.get("content") or "").strip()
        L = len(c)
        if L < 200 or L > 20000:
            continue
        # 【记】去掉维基标记残留过多的条目（信息框/分类堆积）
        if c.count("Category:") > 10 or c.count("{{") > 20:
            continue
        out.append(c)
        total += L
        kept += 1
        if total >= TARGET:
            break

blob = "\n\n".join(out).encode("utf-8")
open(out_path, "wb").write(blob)
print(f"扫描 {n} 条，保留 {kept} 条，共 {len(blob)/1e6:.1f}MB -> {out_path}")
print("抽样:", blob[5000:5180].decode("utf-8", errors="replace"))
