# -*- coding: utf-8 -*-
"""汇总持续学习电池结果：解析 results_*.txt → summary.csv → F̄/P̄ 统计 + 配对t检验。

用法：python summarize_results.py
输出：
  cpp/summary.csv          每行一条 EVAL 记录
  cpp/stats_summary.txt    各组 F̄/P̄ 均值±标准差（K=3）、H1 配对 t 检验
"""
import re, os, csv, glob
from statistics import mean, stdev

ARMS = ["A", "B0", "B1", "B2", "B3", "C", "D"]

def parse():
    rows = []
    pat = re.compile(r"EVAL arm=(\w+) seed=(\d+) after=(\w+) T1=([\d.]+)(?: T2=([\d.]+))?(?: T3=([\d.]+))?")
    for f in glob.glob("results_*.txt"):
        for line in open(f, encoding="utf-8"):
            m = pat.search(line)
            if m:
                rows.append({
                    "arm": m.group(1), "seed": int(m.group(2)), "after": m.group(3),
                    "T1": float(m.group(4)), "T2": float(m.group(5)) if m.group(5) else None,
                    "T3": float(m.group(6)) if m.group(6) else None,
                })
    return rows

def main():
    rows = parse()
    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "seed", "after", "T1", "T2", "T3"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 取每个 (arm, seed) 的最终状态（after 排序：D1<D2<D3）
    order = {"D1": 1, "D2": 2, "D3": 3}
    final = {}
    for r in rows:
        key = (r["arm"], r["seed"])
        if key not in final or order.get(r["after"], 0) > order.get(final[key]["after"], 0):
            final[key] = r

    lines = []
    lines.append("=== 各组最终状态（K=3 均值±标准差）===")
    lines.append(f"{'arm':>4} | {'T1':>16} | {'T2':>16} | {'T3':>16} | {'F̄=(T1d-T1a)+(T2d-T2a)':>22} | {'P̄=T3':>8}")
    f1_data, favg_data, p_data = {}, {}, {}
    for arm in ARMS:
        seeds = sorted(s for (a, s) in final if a == arm)
        if not seeds:
            continue
        # F_i 需要 d1/d2/d3 的 T1/T2 轨迹
        f1s, f2s, pvals = [], [], []
        for s in seeds:
            traj = sorted([r for r in rows if r["arm"] == arm and r["seed"] == s],
                          key=lambda r: order.get(r["after"], 0))
            d = {r["after"]: r for r in traj}
            if "D3" in d and "D1" in d:
                f1 = d["D3"]["T1"] - d["D1"]["T1"]
                f1s.append(f1)
                if "D2" in d:
                    f2s.append(d["D3"]["T2"] - d["D2"]["T2"])
                pvals.append(d["D3"]["T3"])
        favg = [(a + b) / 2 for a, b in zip(f1s, f2s)] if f2s else f1s
        f1_data[arm], favg_data[arm], p_data[arm] = f1s, favg, pvals
        def fmt(xs):
            if not xs:
                return "—"
            return f"{mean(xs):7.4f} ±{stdev(xs):5.4f}" if len(xs) > 1 else f"{xs[0]:7.4f}"
        lines.append(f"{arm:>4} | {fmt(f1s):>16} | {fmt(f2s) if f2s else '—':>16} | {fmt(pvals) if pvals else '—':>16} | {fmt(favg):>22} | {fmt(pvals) if pvals else '—':>8}")

    # H1：配对 t 检验（B1 vs A）
    try:
        from scipy import stats as sc
        lines.append("")
        lines.append("=== H1 配对 t 检验（各 B 组 vs A，F̄ 差值）===")
        for arm in ["B0", "B1", "B2", "B3"]:
            if arm in favg_data and "A" in favg_data:
                common = sorted(set(range(1, 4)) & set(int(s) for (a, s) in final if a == arm)
                                & set(int(s) for (a, s) in final if a == "A"))
                if len(common) >= 2:
                    xa = [favg_data["A"][i - 1] for i in common if i - 1 < len(favg_data["A"])]
                    xb = [favg_data[arm][i - 1] for i in common if i - 1 < len(favg_data[arm])]
                    t, p = sc.ttest_rel(xb, xa)
                    lines.append(f"B1-type {arm} vs A: t={t:.3f} p={p:.4f} (n={len(common)})")
    except ImportError:
        lines.append("(scipy 不可用，跳过 t 检验)")

    txt = "\n".join(lines)
    with open("stats_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

if __name__ == "__main__":
    main()
