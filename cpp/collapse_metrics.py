# -*- coding: utf-8 -*-
"""坍缩实验指标计算 + 下一代语料拼接。

用法：
  python collapse_metrics.py <gen文件> <代编号k>     # 计算 UTF-8 合法率、字频 KL、自损失留档
  python collapse_metrics.py stitch <gen文件>        # 拼 corpus_{k+1} = gen_k + t2 前 100KB

指标写入 collapse_metrics.csv（追加）。
字频 KL：对可解码文本按「字符」统计频率，与 t1_train.bin 的字符频率比较（t1=四大名著）。
"""
import sys, os, csv
from collections import Counter

T2_HEAD = 100 * 1024  # 混入的真实字节数

def utf8_legality(data: bytes):
    """逐字节扫描，统计「合法 UTF-8 前缀切分」比例（宽松版）。"""
    i, n = 0, len(data)
    ok, total = 0, 0
    while i < n:
        b = data[i]
        total += 1
        if b < 0x80:
            ok += 1
            i += 1
        elif 0xC2 <= b <= 0xDF and i + 1 < n and 0x80 <= data[i+1] <= 0xBF:
            ok += 1
            i += 2
        elif 0xE0 <= b <= 0xEF and i + 2 < n and all(0x80 <= data[i+j] <= 0xBF for j in (1, 2)):
            ok += 1
            i += 3
        elif 0xF0 <= b <= 0xF4 and i + 3 < n and all(0x80 <= data[i+j] <= 0xBF for j in (1, 2, 3)):
            ok += 1
            i += 4
        else:
            i += 1  # 非法起始字节，跳过
    return ok / max(total, 1)

def char_freq(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    c = Counter(text)
    total = sum(c.values()) or 1
    return {ch: cnt / total for ch, cnt in c.items()}

def kl_div(p: dict, q: dict):
    import math
    eps = 1e-12
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        if pk > 0:
            s += pk * math.log(pk / max(qk, eps))
    return s

_ref_cache = None
def ref_freq():
    global _ref_cache
    if _ref_cache is None:
        with open("t1_train.bin", "rb") as f:
            _ref_cache = char_freq(f.read(8 * 1024 * 1024))
    return _ref_cache

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "stitch":
        gen_file = sys.argv[2]
        out = "corpus_next.bin"
        with open(gen_file, "rb") as f:
            gen = f.read()
        with open("t2_train.bin", "rb") as f:
            real = f.read(T2_HEAD)
        with open(out, "wb") as f:
            f.write(gen)
            f.write(real)
        print("stitched", out, len(gen) + len(real), "bytes")
        sys.exit(0)

    gen_file, k = sys.argv[1], sys.argv[2]
    with open(gen_file, "rb") as f:
        data = f.read()
    leg = utf8_legality(data)
    kl = kl_div(char_freq(data), ref_freq())
    print(f"G{k}: bytes={len(data)} utf8_legality={leg:.4f} charKL={kl:.4f}")
    exists = os.path.exists("collapse_metrics.csv")
    with open("collapse_metrics.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["gen", "bytes", "utf8_legality", "char_kl_vs_t1"])
        w.writerow([f"G{k}", len(data), f"{leg:.6f}", f"{kl:.6f}"])
