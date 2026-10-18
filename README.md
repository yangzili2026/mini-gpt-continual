# mini-gpt-continual

From-scratch byte-level mini GPT in C++ (862K params, hand-written backpropagation) and continual-learning / self-replay research.

从零开始用 C++ 实现的字节级迷你 GPT（86.2 万参数，手写反向传播），以及持续学习 / 自回放研究。

## About / 简介

A byte-level mini GPT written from scratch in C++ — no ML frameworks, just linear algebra and hand-written backpropagation with Adam. Progression across lessons 1–4:

用 C++ 从零实现的字节级迷你 GPT——不使用任何机器学习框架，只有线性代数、手写反向传播和 Adam 优化器。第 1–4 课的演进路线：

1. **Byte counting** — token/byte statistics over a UTF-8 corpus / 在 UTF-8 语料上做字节统计
2. **Bigram** — a simple 256×256 byte-level bigram model / 简单的 256×256 字节级 bigram 模型
3. **Attention** — single-head causal self-attention from scratch / 从零实现单头因果自注意力
4. **Full GPT** — complete transformer (862K params) with hand-written backpropagation + Adam / 完整 Transformer（86.2 万参数），手写反向传播 + Adam

## Roadmap / 路线图

- Continual-learning experiments: self-replay, replay buffers / 持续学习实验：自回放、回放缓冲区
- Free UTF-8 verifier for generated text / 免费的生成文本 UTF-8 校验器
- More lessons and scaling experiments / 更多课程与规模实验

## Author / 作者

**Zili Yang（杨子立）** — middle-school student researcher / 中学生研究者

Contact: yangzili_2026@qq.com

## Project structure / 项目结构（2026-10-18 快照）

```
cpp/                    全部 C++/CUDA 源码（lesson1–7 + 实验变体）
  lesson1_count.cpp     第1课：字节统计
  lesson2_bigram.cpp    第2课：bigram 模型
  lesson3_attention.cpp 第3课：注意力机制
  lesson4_gpt.cpp       第4课：完整 862K Transformer（论文原型机）
  lesson4_drop.cpp      Dropout 实验版（attout+hid 两处，反向 Dropout）
  lesson4_cn.cpp        全中文 652MB 语料版
  lesson4_small.cpp     小语料（红楼梦30万字节）版
  lesson5_gemm.cu       GPU 矩阵乘法（朴素/瓷砖 kernel）
  lesson5_gpt.cu        GPU 训练版（T=32）
  lesson5_forget.cpp    遗忘统计评测器（字节级热力图数据源）
  lesson6_gpt.cu        视野 128 字节版（GPU）
  lesson7_attres.cpp    注意力残差实验版（Kimi 论文简化复刻）
  gpt*.bin              模型权重存档（用户要求保留，见 .gitignore 备注）
根目录                  语料构建脚本、实验报告、论文草稿、遗忘热力图
```

## Experiment reports / 实验报告

- `实验战役总报告.md` — 三实验汇总 + 模型数据全家福
- `小语料过拟合实验报告.md` — 灾难性遗忘三组对照（论文 §4.1 预实验）
- `注意力残差对比报告.md`、`Dropout对比报告.md` — 结构改造对照实验
- `论文草稿_免费验证器_v4.docx` — 论文最新稿（v4）

## Data sources / 数据来源（大语料不入库，按配方重建）

- FineWeb-Edu sample-10BT（英文）：HuggingFace / ModelScope 镜像
- 天工 skypile 中文网页语料：ModelScope（昆仑万维 Skywork）
- 四大名著、三体：公开文本；唠嗑/学术问答语料：项目内生成脚本生成
- 混合配方见 `build_corpus.py`（原版）与 `build_corpus_cn.py`（全中文加强版）

## Paper / 论文

《没有奖励模型，AI 还能自我复习吗？——字节级微型语言模型的免费验证器、自我回放与逐代坍缩》草稿 v4，全部代码与实验可复现。
