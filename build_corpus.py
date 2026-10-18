# -*- coding: utf-8 -*-
"""把全部语料按配方混合成一个大训练文件 all_corpus.bin（原始字节）"""
import json, os

CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp")

def read(p):
    return open(os.path.join(CPP, p), "rb").read()

def read_jsonl_text(path):
    """提取 jsonl 里的对话内容字段，拼成纯文本"""
    out = []
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
                     encoding="utf-8"):
        r = json.loads(line)
        out.append(r["对话内容"])
    return ("\n\n".join(out)).encode("utf-8")

# (名字, 字节内容, 份数)
recipe = [
    ("四大名著",   b"\n\n".join(read(f) for f in
        ["hongloumeng.txt", "sanguoyanyi.txt", "shuihuzhuan.txt", "xiyouji.txt"]), 6),
    ("天工现代文", read("zh_modern.txt"), 1),
    ("英文FineWeb", read("train_text.txt"), 20),
    ("三体",       read("santi1.txt") + b"\n\n" + read("santi3.txt"), 10),
    ("唠嗑语料",   read_jsonl_text("日常唠嗑语料_带评分.jsonl"), 60),
    ("学术问答",   read_jsonl_text("学术讲评论文问答语料_带评分.jsonl"), 50),
]

parts, total = [], 0
print("混合配方：")
for name, blob, copies in recipe:
    parts.append(blob * copies)     # 内容重复 copies 份
    total += len(blob) * copies
    print(f"  {name}: {len(blob)/1e6:.2f}MB × {copies} = {len(blob)*copies/1e6:.1f}MB")

open(os.path.join(CPP, "all_corpus.bin"), "wb").write(b"\n\n".join(parts))
print(f"合计 {total/1e6:.1f}MB -> cpp/all_corpus.bin")
