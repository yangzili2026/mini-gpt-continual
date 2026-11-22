# -*- coding: utf-8 -*-
"""论文 v4 → v5 第一阶段填充：参考文献全部核实条目 + 硬件信息 + 协议数字。
所有替换均基于 2026-11-22 实际核实的 arXiv/会议页面。
"""
import re, shutil
from docx import Document

SRC = "论文草稿_免费验证器_v4.docx"
DST = "论文草稿_免费验证器_v5_填充中.docx"
shutil.copy(SRC, DST)
d = Document(DST)

# (旧文本, 新文本) 段落级替换
PARA_REPL = [
    # 参考文献完整条目
    ("7. ⟦2605.26097⟧ Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay. 2026.（摘要已于 2026-10 经 arXiv 页面核对）",
     "7. Marek, P., et al. Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay. arXiv:2605.26097, 2026."),
    ("8. ⟦2605.29495⟧ On-Policy Replay for Continual Supervised Fine-Tuning. 2026.（同上，已核对）",
     "8. Chen, Z., et al. On-Policy Replay for Continual Supervised Fine-Tuning. arXiv:2605.29495, 2026."),
    ("9. ⟦2609.32771⟧ CPLUS: Self-Probing for Continual Learning. 2026.（同上，已核对）",
     "9. Cho, J., et al. Continual Learning via Self-Probe Gradients (CPLUS). arXiv:2609.32771, 2026."),
    ("10. ⟦2606.03979⟧ Sleep: Knowledge Seeding and Dreaming. 2026.（同上，已核对）",
     "10. Behrouz, A., et al. Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories. arXiv:2606.03979, 2026."),
    ("11. ⟦2303.10725⟧ SIESTA: Efficient Online Continual Learning with Sleep. 2023.",
     "11. Harun, A., et al. SIESTA: Efficient Online Continual Learning with Sleep. arXiv:2303.10725, 2023."),
    ("12. ⟦2607.17944⟧ The Art of Not Forgetting (CMP). 2026.（同上，已核对）",
     "12. Atmuri, A., Kumar, A., et al. The Art of Not Forgetting (CMP). arXiv:2607.17944, 2026."),
    ("13. ⟦NAACL 2024⟧ Self-generated Replay Memories for Continual NMT.",
     "13. Resta, M., & Bacciu, D. Self-generated Replay Memories for Continual NMT. NAACL 2024. (arXiv:2403.13130)"),
    ("14. ⟦ACL 2024⟧ Self-Synthesized Rehearsal (SSR).",
     "14. Huang, W., et al. Self-Synthesized Rehearsal (SSR). ACL 2024. (arXiv:2403.01244)"),
    ("15. ⟦2502.09963⟧ Generating on Generated: Self-Evolving Diffusion Models. 2025.",
     "15. Zhang, R., et al. Generating on Generated: An Approach Towards Self-Evolving Diffusion Models. arXiv:2502.09963, 2025."),
    ("17. ⟦FineWeb / 天工中文语料出处⟧",
     "17. Penedo, G., et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv:2406.17557, 2024. / Wei, T., et al. Skywork: A More Open Bilingual Foundation Model (SkyPile-150B 中文语料). arXiv:2310.19341, 2023."),
]

# 全局子串替换（在段落文本内做，保留其余文字）
SUBSTR_REPL = [
    ("⟦2605.26097⟧", "Marek et al. (2026)"),
    ("⟦2605.29495⟧", "Chen et al. (2026)"),
    ("⟦2609.32771⟧", "Cho et al. (2026)"),
    ("⟦2606.03979⟧", "Behrouz et al. (2026)"),
    ("⟦2303.10725⟧", "Harun et al. (2023)"),
    ("⟦2607.17944⟧", "Atmuri et al. (2026)"),
    ("⟦2502.09963⟧", "Zhang et al. (2025)"),
    ("⟦NAACL 2024⟧", "(Resta & Bacciu, 2024)"),
    ("⟦ACL 2024⟧", "(Huang et al., 2024)"),
    ("⟦De Lange et al., 2021⟧", "De Lange et al. (2021)"),
    ("⟦占位：CPU 具体型号⟧", "Intel Core i9-14900HX"),
    ("⟦N⟧", "7"),   # 7 组策略：A/B0/B1/B2/B3/C/D
    ("⟦M⟧", "6"),   # 6 代递归：G₀→G₆
    ("⟦K⟧", "3"),   # 3 个随机种子
]

def set_para_text(p, new_text):
    """整段替换文字，沿用首个 run 的字体格式。"""
    if p.runs:
        p.runs[0].text = new_text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(new_text)

n_para = n_sub = 0
exact = set(old for old, _ in PARA_REPL)
for p in d.paragraphs:
    t = p.text
    if t in exact:
        for old, new in PARA_REPL:
            if t == old:
                set_para_text(p, new)
                n_para += 1
                break
        continue
    if "⟦" in t:
        new_t = t
        for old, new in SUBSTR_REPL:
            new_t = new_t.replace(old, new)
        if new_t != t:
            set_para_text(p, new_t)
            n_sub += 1

d.save(DST)
print(f"paragraph-level fills: {n_para}, substring fills: {n_sub}")
print("saved", DST)

# 复查剩余槽位
d2 = Document(DST)
pat = re.compile(r"⟦[^⟧]*⟧")
left = []
for i, p in enumerate(d2.paragraphs):
    for m in pat.finditer(p.text):
        left.append(f"[P{i:03d}] {m.group(0)}")
for t in d2.tables:
    for row in t.rows:
        for cell in row.cells:
            for m in pat.finditer(cell.text):
                left.append(f"[T] {m.group(0)}")
print(f"remaining slots: {len(left)}")
for s in left:
    print(s)
