# -*- coding: utf-8 -*-
# 从 lesson4_drop.cpp 生成 lesson7_attres.cpp（注意力残差版）
# 设计：每个子层入口加"可学习伪查询" qA/qF，对残差流历史快照做 softmax 选择，
# 经零初始化门控 gate 残差加回 x。快照当常数（教学简化），只训练 q 和 gate。
import pathlib

src = pathlib.Path(__file__).parent / "lesson4_drop.cpp"
dst = pathlib.Path(__file__).parent / "lesson7_attres.cpp"
code = src.read_text(encoding="utf-8")

PATCHES = []

# ---- 1. Layer 参数：加 qA/qF/gateA/gateF ----
PATCHES.append((
    "                  W2 = vector<float>(F * D), b2 = vector<float>(D);\n"
    "    // 梯度",
    "                  W2 = vector<float>(F * D), b2 = vector<float>(D),\n"
    "                  // 【记】注意力残差：可学习伪查询 + 零初始化门控（ReZero 式，起点=原模型）\n"
    "                  qA = vector<float>(D), qF = vector<float>(D),\n"
    "                  gateA = vector<float>(1), gateF = vector<float>(1);\n"
    "    // 梯度",
))

# ---- 2. Layer 梯度 ----
PATCHES.append((
    "                  g_W2 = vector<float>(F * D), g_b2 = vector<float>(D);\n"
    "    // 前向缓存",
    "                  g_W2 = vector<float>(F * D), g_b2 = vector<float>(D),\n"
    "                  g_qA = vector<float>(D), g_qF = vector<float>(D),\n"
    "                  g_gateA = vector<float>(1), g_gateF = vector<float>(1);\n"
    "    // 前向缓存",
))

# ---- 3. Layer 前向缓存：加 alpha / means ----
PATCHES.append((
    "                  m_att, m_hid;   // 【记】Dropout 前向掩码，反向传播复用",
    "                  m_att, m_hid,   // 【记】Dropout 前向掩码，反向传播复用\n"
    "                  aA, aF,         // 【记】注意力残差：各快照 softmax 权重（≤ L+1 个）\n"
    "                  mA, mF;         // 【记】注意力残差：各快照均值（当\"键\"用）",
))

# ---- 4. Layer 构造函数 ----
PATCHES.append((
    "        m_att.resize(BT * D); m_hid.resize(BT * F);\n    }",
    "        m_att.resize(BT * D); m_hid.resize(BT * F);\n"
    "        aA.resize(L + 1); aF.resize(L + 1);\n"
    "        mA.resize((size_t)(L + 1) * D); mF.resize((size_t)(L + 1) * D);\n"
    "    }",
))

# ---- 5. zeroGrad ----
PATCHES.append((
    "                       &g_Wo, &g_bo, &g_ln2g, &g_ln2b, &g_W1, &g_b1, &g_W2, &g_b2})",
    "                       &g_Wo, &g_bo, &g_ln2g, &g_ln2b, &g_W1, &g_b1, &g_W2, &g_b2,\n"
    "                       &g_qA, &g_qF, &g_gateA, &g_gateF})",
))

# ---- 6. GPT 结构体：加速照容器 ----
PATCHES.append((
    "    vector<Layer> layers{L};",
    "    vector<Layer> layers{L};\n"
    "    vector<vector<float>> snaps;   // 【记】残差流快照：嵌入后1份 + 每层结束后1份，共 L+1 份"
))
PATCHES.append((
    "        yf.resize(BT * D); probs.resize((size_t)BT * V);\n    }",
    "        yf.resize(BT * D); probs.resize((size_t)BT * V);\n"
    "        snaps.resize(L + 1);\n"
    "        for (auto& s : snaps) s.resize(BT * D);\n"
    "    }",
))

# ---- 7. 参数量统计 ----
PATCHES.append((
    "+ l.ln2g.size() * 2 + l.W1.size() + l.b1.size() + l.W2.size() + l.b2.size();",
    "+ l.ln2g.size() * 2 + l.W1.size() + l.b1.size() + l.W2.size() + l.b2.size()\n"
    "            + l.qA.size() + l.qF.size() + l.gateA.size() + l.gateF.size();",
))

# ---- 8. 初始化 ----
PATCHES.append((
    "        fill(l.b1.begin(), l.b1.end(), 0.0f); fill(l.b2.begin(), l.b2.end(), 0.0f);\n    }",
    "        fill(l.b1.begin(), l.b1.end(), 0.0f); fill(l.b2.begin(), l.b2.end(), 0.0f);\n"
    "        filln(l.qA); filln(l.qF);   // 【记】gate 已是 0，起步时注意力残差=不存在，训练更稳\n"
    "    }",
))

# ---- 9. 插入注意力残差前向/反向函数 ----
PATCHES.append((
    "// ============================================================\n// 前向：toks 是 B×T 个字节编号，返回平均损失",
    """// ============================================================
// 【难】注意力残差（Kimi"每层可选看前面任意层"思想的教学简化版）
// 每个子层入口放一根可学习"伪查询" q：对残差流的所有历史快照
// （嵌入输出 + 每层结束时的 x）算 softmax 权重，把选中的旧信息经门控加回 x。
// 【易错】快照一律当常数（教学简化），反向只训练 q 和 gate，残差主路梯度不动。
// ============================================================
void attres_fwd(GPT& g, Layer& l, int nSnap, bool isAtt,
                const vector<float>& q, const vector<float>& gate) {
    auto& alpha = isAtt ? l.aA : l.aF;
    auto& means = isAtt ? l.mA : l.mF;
    alpha.assign(nSnap, 0.0f);
    means.assign((size_t)nSnap * D, 0.0f);
    // 每份快照求均值，当作"键"
    for (int j = 0; j < nSnap; j++)
        for (int i = 0; i < BT; i++)
            for (int d = 0; d < D; d++)
                means[(size_t)j * D + d] += g.snaps[j][(size_t)i * D + d] / BT;
    // 打分 q·均值 / √D，再做 softmax
    float mx = -1e30f;
    for (int j = 0; j < nSnap; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) s += q[d] * means[(size_t)j * D + d];
        alpha[j] = s / sqrtf((float)D);
        mx = max(mx, alpha[j]);
    }
    float ssum = 0;
    for (int j = 0; j < nSnap; j++) { alpha[j] = expf(alpha[j] - mx); ssum += alpha[j]; }
    for (int j = 0; j < nSnap; j++) alpha[j] /= ssum;
    // 按权重混合快照，经门控残差加回 x
    for (int i = 0; i < BT; i++)
        for (int d = 0; d < D; d++) {
            float mix = 0;
            for (int j = 0; j < nSnap; j++) mix += alpha[j] * g.snaps[j][(size_t)i * D + d];
            g.x[(size_t)i * D + d] += gate[0] * mix;
        }
}

void attres_bwd(GPT& g, Layer& l, int nSnap, bool isAtt,
                const vector<float>& gate, vector<float>& g_q, vector<float>& g_gate,
                const vector<float>& dx) {
    auto& alpha = isAtt ? l.aA : l.aF;
    auto& means = isAtt ? l.mA : l.mF;
    // 1. dα_j = Σ 点积(dx 行, 快照_j 行)；因 x += gate·mix，dmix = gate·dx，先乘进门控
    float dalpha[L + 1];
    float gt = gate[0];
    for (int j = 0; j < nSnap; j++) {
        double s = 0;
        for (int i = 0; i < BT * D; i++) s += (double)dx[i] * g.snaps[j][i];
        dalpha[j] = (float)(gt * s);
    }
    // 2. softmax 反传：dscore_j = α_j · (dα_j - Σ_k α_k·dα_k)，再乘快照均值/√D 进 g_q
    float dot = 0;
    for (int j = 0; j < nSnap; j++) dot += alpha[j] * dalpha[j];
    for (int j = 0; j < nSnap; j++) {
        float dscore = alpha[j] * (dalpha[j] - dot);
        for (int d = 0; d < D; d++)
            g_q[d] += dscore * means[(size_t)j * D + d] / sqrtf((float)D);
    }
    // 3. 门控梯度：d(gate·mix)/dgate = Σ dx·mix
    double gs = 0;
    for (int i = 0; i < BT; i++)
        for (int d = 0; d < D; d++) {
            float mix = 0;
            for (int j = 0; j < nSnap; j++) mix += alpha[j] * g.snaps[j][(size_t)i * D + d];
            gs += (double)dx[(size_t)i * D + d] * mix;
        }
    g_gate[0] += (float)gs;
}

// ============================================================
// 前向：toks 是 B×T 个字节编号，返回平均损失""",
))

# ---- 10. 前向：嵌入后存第 0 份快照 ----
PATCHES.append((
    "    // 2. 每一层\n    static vector<float> attout;",
    "    // 【难】注意力残差：嵌入输出就是第 0 份快照\n"
    "    copy(g.x.begin(), g.x.end(), g.snaps[0].begin());\n"
    "\n    // 2. 每一层\n    static vector<float> attout;",
))

# ---- 11. 层循环改下标式 + 注意力入口注入 ----
PATCHES.append((
    "    for (auto& l : g.layers) {\n"
    "        copy(g.x.begin(), g.x.end(), l.x_in.begin());",
    "    for (int li = 0; li < L; li++) {\n"
    "        auto& l = g.layers[li];\n"
    "        copy(g.x.begin(), g.x.end(), l.x_in.begin());\n"
    "        attres_fwd(g, l, li + 1, true, l.qA, l.gateA);   // 【难】注意力入口：软选择旧快照",
))

# ---- 12. FFN 入口注入 ----
PATCHES.append((
    "        // 前馈网络 + 残差\n"
    "        copy(g.x.begin(), g.x.end(), l.x_in2.begin());",
    "        // 前馈网络 + 残差\n"
    "        copy(g.x.begin(), g.x.end(), l.x_in2.begin());\n"
    "        attres_fwd(g, l, li + 1, false, l.qF, l.gateF);   // 【难】FFN 入口：软选择旧快照",
))

# ---- 13. 层结束存快照 ----
PATCHES.append((
    "        for (int i = 0; i < BT * D; i++) g.x[i] += l.ffn[i];\n    }",
    "        for (int i = 0; i < BT * D; i++) g.x[i] += l.ffn[i];\n"
    "        copy(g.x.begin(), g.x.end(), g.snaps[li + 1].begin());   // 【记】本层结束，存一份快照\n"
    "    }",
))

# ---- 14. 反向：FFN 支路完成后注入点的梯度 ----
PATCHES.append((
    "        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];   // 残差：dx 变为 d(x_in2)\n"
    "\n        // --- 注意力支路 ---",
    "        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];   // 残差：dx 变为 d(x_in2)\n"
    "        attres_bwd(g, l, li + 1, false, l.gateF, l.g_qF, l.g_gateF, dx);   // 【难】FFN 注入点\n"
    "\n        // --- 注意力支路 ---",
))

# ---- 15. 反向：注意力支路完成后注入点的梯度 ----
PATCHES.append((
    "        layernorm_bwd(l.xhat1.data(), l.rstd1.data(), l.ln1g.data(), dy.data(),\n"
    "                      tmp.data(), l.g_ln1g.data(), l.g_ln1b.data(), BT);\n"
    "        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];\n    }",
    "        layernorm_bwd(l.xhat1.data(), l.rstd1.data(), l.ln1g.data(), dy.data(),\n"
    "                      tmp.data(), l.g_ln1g.data(), l.g_ln1b.data(), BT);\n"
    "        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];\n"
    "        attres_bwd(g, l, li + 1, true, l.gateA, l.g_qA, l.g_gateA, dx);   // 【难】注意力注入点\n"
    "    }",
))

# ---- 16. Adam：新参数入队 ----
PATCHES.append((
    "        a.step(l.W2, l.g_W2, t, lr); a.step(l.b2, l.g_b2, t, lr);\n    }",
    "        a.step(l.W2, l.g_W2, t, lr); a.step(l.b2, l.g_b2, t, lr);\n"
    "        a.step(l.qA, l.g_qA, t, lr); a.step(l.qF, l.g_qF, t, lr);\n"
    "        a.step(l.gateA, l.g_gateA, t, lr); a.step(l.gateF, l.g_gateF, t, lr);\n"
    "    }",
))

# ---- 17. 读档 ----
PATCHES.append((
    "                rd(l.ln2g); rd(l.ln2b); rd(l.W1); rd(l.b1); rd(l.W2); rd(l.b2);",
    "                rd(l.ln2g); rd(l.ln2b); rd(l.W1); rd(l.b1); rd(l.W2); rd(l.b2);\n"
    "                rd(l.qA); rd(l.qF); rd(l.gateA); rd(l.gateF);",
))

# ---- 18. 存档 ----
PATCHES.append((
    "                wr(l.ln2g); wr(l.ln2b); wr(l.W1); wr(l.b1); wr(l.W2); wr(l.b2);",
    "                wr(l.ln2g); wr(l.ln2b); wr(l.W1); wr(l.b1); wr(l.W2); wr(l.b2);\n"
    "                wr(l.qA); wr(l.qF); wr(l.gateA); wr(l.gateF);",
))

for i, (old, new) in enumerate(PATCHES, 1):
    n = code.count(old)
    assert n == 1, f"补丁{i} 锚点出现 {n} 次（应为1次）"
    code = code.replace(old, new)

dst.write_text(code, encoding="utf-8")
print("OK ->", dst)
