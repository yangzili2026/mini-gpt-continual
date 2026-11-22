# -*- coding: utf-8 -*-
"""论文 v5 第二阶段填充（临时结果段）：把 2026-11-22 中期真实数据写进 5.1 节。
明确标注 interim，K=3 全部跑完 + 配对 t 检验后再改写为正式结论。
"""
from docx import Document
from docx.shared import RGBColor

DST = "论文草稿_免费验证器_v5_填充中.docx"
d = Document(DST)

anchor = None
for p in d.paragraphs:
    if p.text.strip().startswith("5.2 RQ2"):
        anchor = p
        break
assert anchor is not None, "找不到 5.2 节标题"

paras = [
    ("【临时结果 · 2026-11-22 · interim】以下数字来自已完成的 9 组实验"
     "（B0/B1/C 各 3 种子、B2 s1、A s1），A 组 s2/s3 与 B2/B3/D 组仍在训练，"
     "正式判定以 K=3 全量 + 配对 t 检验为准。", True),
    ("不复习的对照组 A 表现出典型灾难性遗忘：学完三个域后 T1（四大名著）测试损失"
     "从 2.09 升至 6.73（遗忘 +4.63 nats），T2（天工现代文）从 2.03 升至 6.21。", False),
    ("三类自生成回放均显著缓解遗忘但彼此接近：B0（原始自回放）T1 终值 5.05，"
     "B1（UTF-8 过滤）5.12，B2（自困惑度过滤，暂为单种子）5.36——三种过滤信号的差异"
     "在种子噪声范围之内。原因初步定位为：经 G₀ 预训练后，模型自生成字节的 UTF-8 合法率"
     "已达约 99.6%，免费验证器可拦截的「垃圾」样本所剩无几，过滤器的边际作用自然有限。", False),
    ("真实数据回放（组 C）展示了上界的真正位置：T1 终值 2.30，遗忘仅 +0.20，"
     "几乎完全不遗忘，且 T3 最终损失（0.22）与 B 组相当。"
     "自回放与真实回放之间约 2.8 nats 的遗忘差距，量化了免费验证器替代奖励模型的真实边界："
     "它能防止崩盘，但还远不能替代真实数据。", False),
    ("对 H1 的临时判定：自回放显著优于 A（方向明确，幅度约 1.6–1.7 nats），"
     "但「F̄ 达到上界某一比例即视为替代成立」的阈值⟦x⟧需要结合全部数据后再定；"
     "若按「达到组 C 的 50%」衡量，自回放（F̄≈2.9）暂不满足，H1 倾向部分支持——"
     "免费验证器有效但不足以单独替代真实回放。", False),
]

for text, is_note in paras:
    p = anchor.insert_paragraph_before(text)
    if is_note:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
            r.font.bold = True

d.save(DST)
print("saved", DST, "插入了", len(paras), "段临时结果")
