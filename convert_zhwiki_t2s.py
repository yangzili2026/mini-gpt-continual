# -*- coding: utf-8 -*-
"""zhwiki 繁转简 v2：按字节偏移续跑（seek 跳过已转部分，不重复解码）。
反复运行直到输出 DONE。每轮限时 240 秒。"""
import os, time
from opencc import OpenCC

SRC = "cpp/zhwiki_cn.txt"
DST = "cpp/zhwiki_cn_simplified.txt"
STATE = "convert_t2s.state"
BUDGET = 240

cc = OpenCC("t2s")
offset = int(open(STATE).read().strip()) if os.path.exists(STATE) else 0
mode = "a" if offset else "w"
print(f"从字节 {offset} 继续")

t0 = time.time()
done = 0
with open(SRC, encoding="utf-8") as fin, open(DST, mode, encoding="utf-8") as fout:
    fin.seek(offset)
    while time.time() - t0 < BUDGET:
        line = fin.readline()
        if not line:
            offset = fin.tell()
            open(STATE, "w").write(str(offset))
            os.remove(STATE)
            print(f"✅ DONE {DST} = {os.path.getsize(DST)/1e6:.1f}MB（源 {os.path.getsize(SRC)/1e6:.1f}MB 全部转完）")
            break
        fout.write(cc.convert(line))
        done += 1
        if done % 20000 == 0:
            offset = fin.tell()
    else:
        offset = fin.tell()
        open(STATE, "w").write(str(offset))
        total = os.path.getsize(SRC)
        print(f"本轮转换 {done} 行，进度 {offset/1e6:.0f}/{total/1e6:.0f}MB，再运行本脚本续转")
