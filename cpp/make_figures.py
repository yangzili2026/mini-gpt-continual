# -*- coding: utf-8 -*-
"""论文配图一键生成（数据齐后运行）。
图2 遗忘曲线 / 图3 F̄-P̄权衡散点 / 图4 逐代坍缩三线 / 图5 逐层梯度冲突。
用法：python make_figures.py [图号，缺省全部]
"""
import sys, os, csv, re, glob
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import matplotlib
matplotlib.use("Agg")
setup_plot()
import matplotlib.pyplot as plt

WS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WS, "figures")
os.makedirs(OUT, exist_ok=True)

ARMS = ["A", "B0", "B1", "B2", "B3", "C", "D"]
ARM_LABEL = {"A": "A no-replay", "B0": "B0 raw self", "B1": "B1 UTF-8 filter",
             "B2": "B2 self-ppl filter", "B3": "B3 n-gram filter",
             "C": "C real replay (upper)", "D": "D grad surgery"}
ARM_COLOR = {"A": "#888888", "B0": "#cc6600", "B1": "#0066cc", "B2": "#009933",
             "B3": "#9900cc", "C": "#00aaaa", "D": "#cc0033"}
ORDER = {"D1": 1, "D2": 2, "D3": 3}

def load_summary():
    rows = []
    pat = re.compile(r"EVAL arm=(\w+) seed=(\d+) after=(\w+) T1=([\d.]+)(?: T2=([\d.]+))?(?: T3=([\d.]+))?")
    for f in glob.glob(os.path.join(WS, "results_*.txt")):
        for line in open(f, encoding="utf-8"):
            m = pat.search(line)
            if m:
                rows.append({"arm": m.group(1), "seed": int(m.group(2)),
                             "after": m.group(3), "T1": float(m.group(4)),
                             "T2": float(m.group(5)) if m.group(5) else None,
                             "T3": float(m.group(6)) if m.group(6) else None})
    return rows

def fig2(rows):
    """图2：遗忘曲线——T1/T2 测试损失随训练阶段推进，各组均值（K=3）。"""
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["after"] in ORDER:
            data[r["arm"]][r["after"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ti, key in enumerate(["T1", "T2"]):
        ax = axes[ti]
        for arm in ARMS:
            if arm not in data:
                continue
            ys, es = [], []
            for stage in ["D1", "D2", "D3"]:
                vals = [r[key] for r in data[arm][stage] if r[key] is not None]
                if stage == "D1" and key == "T2":
                    vals = []
                if vals:
                    m = sum(vals) / len(vals)
                    sd = (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0
                else:
                    m, sd = float("nan"), 0.0
                ys.append(m); es.append(sd)
            ax.errorbar([1, 2, 3], ys, yerr=es, marker="o", capsize=3,
                        label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["after D1", "after D2", "after D3"])
        task_name = ["T1 Four Classics", "T2 Tianggong modern zh"][ti]
        ax.set_title(f"Test loss on {task_name}")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig.2 Forgetting curves: sequential learning T1→T2→T3 (mean ± range, K=3)")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_forgetting_curves.png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved", p)

def fig3(rows):
    """图3：F̄-P̄ 权衡散点。"""
    from statistics import mean
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        per[r["arm"]][r["seed"]][r["after"]] = r
    pts = []
    for arm in ARMS:
        f1s, f2s, pvals = [], [], []
        for seed, d in per[arm].items():
            if "D3" in d and "D1" in d:
                f1s.append(d["D3"]["T1"] - d["D1"]["T1"])
                if "D2" in d and d["D2"]["T2"] is not None:
                    f2s.append(d["D3"]["T2"] - d["D2"]["T2"])
                pvals.append(d["D3"]["T3"])
        if not f1s:
            continue
        favg = [(a + b) / 2 for a, b in zip(f1s, f2s)] if len(f2s) == len(f1s) else f1s
        pts.append((mean(pvals), mean(favg), arm))
    fig, ax = plt.subplots(figsize=(6, 5))
    for p, f, arm in pts:
        ax.scatter(p, f, s=120, color=ARM_COLOR[arm], zorder=3)
        ax.annotate(ARM_LABEL[arm], (p, f), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("P̄ (final loss on T3, lower=better)")
    ax.set_ylabel("F̄ (average forgetting, lower=better)")
    ax.set_title("Fig.3 Replay-signal trade-off: forgetting vs final performance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_tradeoff.png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved", p)

def fig4():
    """图4：逐代坍缩三线（UTF-8 合法率 / 字频 KL / 自损失）。"""
    csv_path = os.path.join(WS, "collapse_metrics.csv")
    if not os.path.exists(csv_path):
        print("skip fig4: no collapse_metrics.csv")
        return
    gens, leg, kl, selfloss = [], [], [], []
    sl_file = os.path.join(WS, "collapse_selfloss.csv")
    sl_map = {}
    if os.path.exists(sl_file):
        for row in csv.DictReader(open(sl_file, encoding="utf-8")):
            sl_map[row["gen"]] = float(row["self_loss"])
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        g = row["gen"].replace("G", "")
        gens.append(int(g))
        leg.append(float(row["utf8_legality"]))
        kl.append(float(row["char_kl_vs_t1"]))
        selfloss.append(sl_map.get(row["gen"], float("nan")))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specs = [(leg, "UTF-8 legality rate", "Fig.4a"), (kl, "char-frequency KL vs T1", "Fig.4b"),
             (selfloss, "self-loss (eval on own gen data)", "Fig.4c")]
    for ax, (ys, ylab, ttl) in zip(axes, specs):
        ax.plot(gens, ys, marker="o", color="#0066cc")
        ax.set_xticks(gens); ax.set_xticklabels([f"G{g}" for g in gens])
        ax.set_ylabel(ylab); ax.set_title(ttl); ax.grid(alpha=0.3)
    fig.suptitle("Fig.4 Recursive self-training collapse, G0→G6")
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_collapse.png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved", p)

def fig5():
    """图5：D 组逐层梯度冲突强度 cos 分布。"""
    files = glob.glob(os.path.join(WS, "surg_D_s*.txt"))
    if not files:
        print("skip fig5: no surg_D_s*.txt")
        return
    layers = defaultdict(list)
    for f in files:
        for line in open(f, encoding="utf-8"):
            m = re.match(r"t=(\d+)\s+(.*)", line.strip())
            if not m:
                continue
            for kv in m.group(2).split():
                k, v = kv.split("=")
                if k.startswith("g"):
                    layers[int(k[1:])].append(float(v))
    names = ["embed", "layer1", "layer2", "layer3", "layer4", "head"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = sorted(layers)
    data = [layers[x] for x in xs]
    ax.boxplot(data, labels=[names[x] if x < len(names) else f"g{x}" for x in xs])
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="cos=0 (conflict threshold)")
    ax.set_ylabel("cos(g_new, g_probe)")
    ax.set_title("Fig.5 Per-layer gradient conflict intensity (arm D, all steps × seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_layer_conflict.png")
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved", p)

if __name__ == "__main__":
    which = sys.argv[1:] or ["2", "3", "4", "5"]
    rows = load_summary()
    if "2" in which:
        fig2(rows)
    if "3" in which:
        fig3(rows)
    if "4" in which:
        fig4()
    if "5" in which:
        fig5()
