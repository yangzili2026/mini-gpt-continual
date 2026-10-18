import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup_plot()
fig, ax = plt.subplots(figsize=(12, 13.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 132); ax.axis('off')

GREEN = '#2e9e5b'; GREEN_F = '#d8f2e1'
YELLOW = '#d9a400'; YELLOW_F = '#fdf2c9'
RED = '#cc2b2b'; RED_F = '#f9d7d7'
GRAY = '#555555'

def box(x, y, w, h, text, edge, face, fs=9, bold=False, tc='black'):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                       linewidth=2, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color=tc)

def arrow(x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=18, linewidth=1.8, color=color))

ax.text(50, 129.5, '我们的迷你 GPT（862K 参数）——现状架构图',
        ha='center', fontsize=17, fontweight='bold')
ax.text(50, 126.8, '仿《Attention Is All You Need》Figure 1 布局 · 颜色 = 完成度 / 紧急度',
        ha='center', fontsize=10, color='#777777')

# ---- 主干：数据流（自上而下） ----
bx, bw = 12, 42
box(bx, 119, bw, 4.5, '语料：179MB 混合语料\n四大名著×6 + 天工 + 英文×20 + 三体×10 + 唠嗑×60 + 学术×50', GREEN, GREEN_F, 8.3)
box(bx, 112.5, bw, 4, '输入窗口：33 个字节（修复后多带 1 个答案字节）', GREEN, GREEN_F, 9)
box(bx, 106.5, bw, 4, '嵌入表 E（256×128）+ 位置编码 P（32×128）', GREEN, GREEN_F, 9)
arrow(bx+bw/2, 119, bx+bw/2, 116.7)
arrow(bx+bw/2, 112.5, bx+bw/2, 110.7)

ax.text(bx+bw/2, 103.8, '×4 层 Transformer 块', ha='center', fontsize=11, fontweight='bold')
arrow(bx+bw/2, 106.5, bx+bw/2, 105.2)

box(bx+2, 92, bw-16, 4, 'LayerNorm', GREEN, GREEN_F, 9)
box(bx+2, 86.5, bw-16, 4, '多头注意力 4头×32维\nQ·K/√32 → 掩码 → softmax → ·V', GREEN, GREEN_F, 8.5)
box(bx+2, 81.5, bw-16, 4, '残差相加', GREEN, GREEN_F, 9)
box(bx+2, 76, bw-16, 4, 'LayerNorm', GREEN, GREEN_F, 9)
box(bx+2, 70.5, bw-16, 4, '前馈网络 128→512→128（ReLU）', GREEN, GREEN_F, 9)
box(bx+2, 65.5, bw-16, 4, '残差相加', GREEN, GREEN_F, 9)
for y1, y2 in [(92, 90.7), (86.5, 85.7), (81.5, 80.2), (76, 74.7), (70.5, 69.7)]:
    arrow(bx+2+(bw-16)/2, y1, bx+2+(bw-16)/2, y2)

ax.add_patch(FancyArrowPatch((bx+bw-11, 98.5), (bx+bw-11, 63.5),
             arrowstyle='-|>', mutation_scale=14, linewidth=1.5,
             color=GRAY, connectionstyle='arc3,rad=0'))
ax.text(bx+bw-9.3, 81, '残差\n×2', fontsize=8, color=GRAY, ha='left', va='center')

box(bx, 56.5, bw, 4, '最终 LayerNorm → 输出头 W_U（128→256）→ softmax', GREEN, GREEN_F, 8.5)
box(bx, 50.5, bw, 4, '交叉熵损失（32 个字位全部批改）', GREEN, GREEN_F, 9)
arrow(bx+bw/2, 65.5, bx+bw/2, 60.7)
arrow(bx+bw/2, 56.5, bx+bw/2, 54.7)

ax.add_patch(FancyArrowPatch((bx-2, 52.5), (5.5, 52.5), arrowstyle='-', linewidth=1.8, color=GRAY))
ax.add_patch(FancyArrowPatch((5.5, 52.5), (5.5, 87), arrowstyle='-', linewidth=1.8, color=GRAY))
ax.add_patch(FancyArrowPatch((5.5, 87), (bx, 87), arrowstyle='-|>', mutation_scale=14, linewidth=1.8, color=GRAY))
ax.text(3.5, 70, '手写反向传播 + Adam（学习率先大后小）', fontsize=8.5,
        color=GRAY, ha='center', va='center', rotation=90)

# ---- 右侧：状态面板 ----
sx, sw = 62, 36
ax.text(sx+sw/2, 123.5, '状 态 面 板', ha='center', fontsize=13, fontweight='bold')

box(sx, 104, sw, 15,
    '已 完 成（绿）\n'
    '· 第1课 字节统计 → 第2课 bigram\n'
    '  → 第3课 注意力+手写反向传播+Adam\n'
    '· 第4课 完整迷你GPT：CPU 训 2400 步\n'
    '  训练损失 2.22（随机水平 5.545）\n'
    '· 期末考：红楼 2.28 / 三体 2.07 / 天工 2.49\n'
    '· 仿写 UTF-8 合法率 99.6%\n'
    '· 双 bug 已修：第32字位盲区 + softmax 缺失',
    GREEN, GREEN_F, 8.3)

box(sx, 88.5, sw, 11,
    '待 办 · 较 急（黄）\n'
    '· 论文训练三件套：\n'
    '  dropout（防背课文）/ warmup（热身）\n'
    '  / label smoothing（别过度自信）\n'
    '· 模型增容：更大 D、更多层 → 句子更通顺',
    YELLOW, YELLOW_F, 8.3)

box(sx, 74.5, sw, 10,
    '阻 塞 · 最 急（红）\n'
    '· GPU 加速：RTX 4070 闲置中！\n'
    '→ 等你双击安装 cuda_network.exe\n'
    '→ 装完我写 CUDA 矩阵乘 kernel\n'
    '→ 训练提速约 10 倍',
    RED, RED_F, 8.3)

box(sx, 56, sw, 14,
    '一 个 半 月 的 路（回顾）\n'
    '9/26  零基础起念，Web Bridge 跑通\n'
    '9/27  下载 10B token 英文语料\n'
    '      （32MB 分块+6线程断点续传器）\n'
    '10/3  改用 C++ 重开四课\n'
    '      GESP3 级基础全程跟上\n'
    '10/4  混入全部语料 + 抓出双 bug\n'
    '      模型正式会写字',
    GRAY, '#eeeeee', 8.3)

# ---- 底部图例 ----
box(16, 4, 16, 4.5, '绿 = 已跑通', GREEN, GREEN_F, 9.5)
box(38, 4, 17, 4.5, '黄 = 待办 · 较急', YELLOW, YELLOW_F, 9.5)
box(61, 4, 17, 4.5, '红 = 阻塞 · 最急', RED, RED_F, 9.5)

out = Path(__file__).parent / '架构图_迷你GPT现状.png'
fig.savefig(out, bbox_inches='tight', dpi=150)
print('saved:', out)
