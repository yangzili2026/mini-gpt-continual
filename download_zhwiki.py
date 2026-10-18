# -*- coding: utf-8 -*-
"""zhwiki 2.46GB 分块断点续传下载器（6线程×32MB块，块级续传，可反复运行直到下完）"""
import os, sys, time, math, threading, requests

URL = "https://modelscope.cn/api/v1/datasets/caoaolong/zhwiki/repo?Revision=master&FilePath=zhwiki_dataset.jsonl"
OUT = "zhwiki_dataset.jsonl"
TMP = OUT + ".chunks"
TOTAL = 2462455277
CHUNK = 32 * 1024 * 1024
N = math.ceil(TOTAL / CHUNK)
THREADS = 6

os.makedirs(TMP, exist_ok=True)
lock = threading.Lock()
done = [False] * N
todo = []

def have(i):
    p = os.path.join(TMP, f"{i:05d}")
    return os.path.exists(p) and os.path.getsize(p) == (min(CHUNK, TOTAL - i * CHUNK))

for i in range(N):
    if have(i):
        done[i] = True
    else:
        todo.append(i)
print(f"总块数 {N}，已完成 {sum(done)}，待下 {len(todo)}")

stop = False
def worker():
    global stop
    while True:
        with lock:
            if not todo or stop:
                return
            i = todo.pop(0)
        start = i * CHUNK
        end = min(start + CHUNK - 1, TOTAL - 1)
        for attempt in range(4):
            try:
                r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                 timeout=(20, 120), stream=True)
                data = b"".join(r.iter_content(1 << 20))
                if len(data) == end - start + 1:
                    with open(os.path.join(TMP, f"{i:05d}"), "wb") as f:
                        f.write(data)
                    with lock:
                        done[i] = True
                    break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            with lock:
                todo.append(i)   # 失败块回队尾，下次运行再试

threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
t0 = time.time()
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(5)
    el = time.time() - t0
    got = sum(done)
    if el > 0 and got:
        spd = got * CHUNK / el / 1e6
        print(f"[{el:.0f}s] 完成 {got}/{N} 块  {spd:.1f} MB/s", flush=True)
    if el > 260:          # 【记】快超时时主动收工，已下块已落盘，重跑即续
        stop = True
        print("TIME_SLICE_END（重跑本脚本续传）")
        break

if all(done):
    with open(OUT, "wb") as f:
        for i in range(N):
            with open(os.path.join(TMP, f"{i:05d}"), "rb") as c:
                f.write(c.read())
    for i in range(N):
        os.remove(os.path.join(TMP, f"{i:05d}"))
    os.rmdir(TMP)
    sz = os.path.getsize(OUT)
    print(f"✅ 合并完成 {OUT} = {sz} 字节（{'一致' if sz == TOTAL else '⚠️大小不符'}）")
