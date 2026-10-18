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
