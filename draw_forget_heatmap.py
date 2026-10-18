# -*- coding: utf-8 -*-
"""AI 遗忘热力图：读 forget_stats.txt，画出三张图"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import matplotlib.pyplot as plt
import numpy as np

setup_plot()

BASE = Path(r"C:\Users\yangz\Documents\Kimi\Workspaces\4小时学习\cpp")
lines = (BASE / "forget_stats.txt").read_text().split()

# ---------- 解析（按空白切词，逐段读）----------
p = 0
def tok():
    global p
    v = lines[p]; p += 1; return v

assert tok() == "POS"
T = int(tok())
pos_loss = np.zeros(T); pos_acc = np.zeros(T)
for t in range(T):
    pos_loss[t] = float(tok()); pos_acc[t] = float(tok())

assert tok() == "BYTE"
assert int(tok()) == 256
byte_loss = np.zeros(256); byte_cnt = np.zeros(256)
for y in range(256):
    byte_loss[y] = float(tok()); byte_cnt[y] = int(tok())

assert tok() == "ATT"
L, H, T2 = int(tok()), int(tok()), int(tok())
att = np.zeros((L, H, T2, T2))
for li in range(L):
    for h in range(H):
        for a in range(T2 * T2):
            att[li, h, a // T2, a % T2] = float(tok())

OUT = Path(r"C:\Users\yangz\Documents\Kimi\Workspaces\4小时学习")

# ---------- 图1：遗忘地图（注意力热力图，2×2 层 + 全局平均）----------
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
avg_all = att.mean(axis=(0, 1))  # 全部层全部头的平均
for li in range(L):
    ax = axes[li // 2, li % 2]
    m = att[li].mean(axis=0)
    im = ax.imshow(m, cmap="hot", vmin=0, vmax=max(0.25, avg_all.max()))
    ax.set_title(f"第 {li+1} 层（4头平均）", fontsize=13)
    ax.set_xlabel("被看的字（越往左=越久远）")
    ax.set_ylabel("正在猜的字")
fig.colorbar(im, ax=axes[0, 0])

ax = axes[0, 2]
im = ax.imshow(avg_all, cmap="hot", vmin=0, vmax=avg_all.max())
ax.set_title("全局平均遗忘地图", fontsize=14, fontweight="bold")
ax.set_xlabel("被看的字（越往左=越久远）")
ax.set_ylabel("正在猜的字")

# 右下：按"距离"的遗忘曲线 —— 距离 d = 正在猜的字 - 被看的字
ax = axes[1, 2]
dist_w = np.zeros(T2); dist_n = np.zeros(T2)
for t in range(T2):
    for j in range(t + 1):
        d = t - j
        dist_w[d] += avg_all[t, j]
        dist_n[d] += 1
ax.plot(range(T2), dist_w / dist_n, "o-", color="crimson", lw=2)
ax.set_title("遗忘曲线：看得越远，记得越少", fontsize=13)
ax.set_xlabel("距离（隔了几个字）")
ax.set_ylabel("平均注意力权重")
ax.grid(alpha=0.3)
axes[1, 0].axis("off"); axes[1, 1].axis("off")
fig.suptitle("AI 遗忘地图 · 字节级注意力热力图（32字节窗口）", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "遗忘地图.png", bbox_inches="tight", dpi=130)
plt.close(fig)

# ---------- 图2：位置损失 + 字节损失 ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
ctx = np.arange(1, T + 1)  # 位置t有 t+1 个字节的上下文
a1.plot(ctx, pos_loss, "o-", color="steelblue", lw=2, label="平均损失")
a1.axhline(5.545, ls="--", color="gray", label="瞎猜水平 5.545")
a1.set_xlabel("上下文长度（能看到前面几个字）")
a1.set_ylabel("平均损失（越低越好）")
a1.set_title("上下文遗忘曲线：窗口里越往后（看得越多）预测越好", fontsize=13)
a1.legend(); a1.grid(alpha=0.3)

# 字节损失：只画出现次数够多的字节，标出 UTF-8 汉字首字节区间
valid = byte_cnt >= 20
xs = np.where(valid)[0]
a2.bar(xs, byte_loss[valid], color="darkorange")
a2.axvspan(0xE0 - 0.5, 0xEF + 0.5, color="red", alpha=0.12)
a2.text(0xE7, max(byte_loss[valid]) * 0.97, "汉字三字节\n首字节区间", ha="center", fontsize=10, color="darkred")
a2.set_xlabel("字节值 0~255")
a2.set_ylabel("平均损失")
a2.set_title("字节级遗忘榜：哪些字节最难猜", fontsize=13)
fig.suptitle("AI 遗忘分析 · 第4课迷你GPT（86万参数）", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "遗忘曲线.png", bbox_inches="tight", dpi=130)
plt.close(fig)

print("saved:", OUT / "遗忘地图.png")
print("saved:", OUT / "遗忘曲线.png")
print("注意力最高权重:", round(float(avg_all.max()), 4), " 最近邻平均:", round(float(dist_w[0]), 4))
print("距离0权重:", round(float((dist_w / dist_n)[0]), 4),
      " 距离31权重:", round(float((dist_w / dist_n)[T2 - 1]), 5))
print("位置0损失:", round(float(pos_loss[0]), 3), " 位置31损失:", round(float(pos_loss[-1]), 3))
worst = np.argsort(-np.where(valid, byte_loss, 0))[:8]
print("最难猜字节:", [(int(y), round(float(byte_loss[y]), 2)) for y in worst])
