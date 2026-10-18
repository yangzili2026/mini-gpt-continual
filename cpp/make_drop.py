# 由 lesson4_gpt.cpp 生成 lesson4_drop.cpp：
# Dropout(反向/inverted, p=0.1) 施加在 attout 和 hid 两处 + 评测期遗忘统计输出
import io

src = io.open('lesson4_gpt.cpp', encoding='utf-8-sig').read()
n0 = len(src)

def rep(old, new, cnt=1):
    global src
    assert src.count(old) >= cnt, 'NOT FOUND: ' + old[:60]
    src = src.replace(old, new, cnt)

# ---- 1. 标题 ----
rep('C++ 造大语言模型 · 第4课：完整迷你 GPT',
    'C++ 造大语言模型 · 第4课+Dropout：完整迷你 GPT')

# ---- 2. 超参数 + Dropout 全局量 ----
rep('const float SQRT_NH = sqrtf((float)NH);',
    '''const float SQRT_NH = sqrtf((float)NH);
const float DROPOUT_P = 0.1f;   // 【记】Dropout 概率：训练时随机关掉 10% 神经元，防背课文
// 【难】反向 Dropout(inverted dropout)：训练时掩码取 0 或 1/(1-p)，评测时什么都不做
mt19937 drop_rng(20261011);                 // Dropout 专用随机数（与取样随机数分开）
uniform_real_distribution<float> drop_ud(0.0f, 1.0f);''')

# ---- 3. Layer 增加两个掩码成员 ----
rep('''    vector<float> x_in, xhat1, rstd1, y1, q, k, v, att, ctx,
                  x_in2, xhat2, rstd2, y2, hid, ffn;''',
    '''    vector<float> x_in, xhat1, rstd1, y1, q, k, v, att, ctx,
                  x_in2, xhat2, rstd2, y2, hid, ffn,
                  m_att, m_hid;   // 【记】Dropout 前向掩码，反向传播复用''')
rep('        hid.resize((size_t)BT * F); ffn.resize(BT * D);',
    '''        hid.resize((size_t)BT * F); ffn.resize(BT * D);
        m_att.resize(BT * D); m_hid.resize(BT * F);''')

# ---- 4. tmp 越界修复（若还没修）----
if 'tmp.resize(BT * D);' in src:
    rep('dx.resize(BT * D); dy.resize(BT * D); tmp.resize(BT * D);',
        'dx.resize(BT * D); dy.resize(BT * D); tmp.resize(BT * F);')

# ---- 5. forward 加 training 参数 ----
rep('float forward(GPT& g, const unsigned char* toks, bool needLast = false) {',
    'float forward(GPT& g, const unsigned char* toks, bool needLast = false, bool training = false) {')

# ---- 6. 注意力输出投影后、加残差前：Dropout ----
rep('''        bias_add(attout.data(), l.bo.data(), BT, D);
        for (int i = 0; i < BT * D; i++) g.x[i] += attout[i];''',
    '''        bias_add(attout.data(), l.bo.data(), BT, D);
        if (training)   // 【易错】反向 Dropout：只在训练时随机关+放大，评测传 false 原样通过
            for (int i = 0; i < BT * D; i++) {
                l.m_att[i] = (drop_ud(drop_rng) < DROPOUT_P) ? 0.0f : 1.0f / (1.0f - DROPOUT_P);
                attout[i] *= l.m_att[i];
            }
        for (int i = 0; i < BT * D; i++) g.x[i] += attout[i];''')

# ---- 7. FFN ReLU 后、W2 前：Dropout ----
rep('''        for (auto& h_ : l.hid) h_ = max(h_, 0.0f);      // ReLU
        gemm(l.hid.data(), l.W2.data(), l.ffn.data(), BT, F, D);''',
    '''        for (auto& h_ : l.hid) h_ = max(h_, 0.0f);      // ReLU
        if (training)
            for (int i = 0; i < BT * F; i++) {
                l.m_hid[i] = (drop_ud(drop_rng) < DROPOUT_P) ? 0.0f : 1.0f / (1.0f - DROPOUT_P);
                l.hid[i] *= l.m_hid[i];
            }
        gemm(l.hid.data(), l.W2.data(), l.ffn.data(), BT, F, D);''')

# ---- 8. backward 加 training 参数 ----
rep('void backward(GPT& g, const unsigned char* toks) {',
    'void backward(GPT& g, const unsigned char* toks, bool training = false) {')

# ---- 9. d(ctx) 乘注意力掩码 ----
rep('        gemm_bt(dx.data(), l.Wo.data(), dctx2.data(), BT, D, D);    // d(ctx)',
    '''        gemm_bt(dx.data(), l.Wo.data(), dctx2.data(), BT, D, D);    // d(ctx)
        if (training)   // 【易错】Dropout 反传：梯度 = 前向掩码 × 上游梯度
            for (int i = 0; i < BT * D; i++) dctx2[i] *= l.m_att[i];''')

# ---- 10. ReLU 反传合并 Dropout 反传 ----
rep('        for (int i = 0; i < BT * F; i++) if (l.hid[i] == 0) tmp[i] = 0;   // ReLU反传',
    '''        // 【难】ReLU+Dropout 联合反传：hid==0 处梯度归 0（含被 Dropout 打掉的），否则乘掩码
        for (int i = 0; i < BT * F; i++)
            tmp[i] = (l.hid[i] == 0) ? 0.0f : tmp[i] * l.m_hid[i];''')

# ---- 11. 训练循环调用点 ----
rep('''        float loss = forward(g, batch.data());
        backward(g, batch.data());''',
    '''        float loss = forward(g, batch.data(), false, true);   // training=true
        backward(g, batch.data(), true);''')

# ---- 12. 评测/仿写调用点传 false（打分逻辑一行不动）----
rep('                forward(g, w.data(), true);',
    '                forward(g, w.data(), true, false);   // training=false：评测不做 Dropout')

# ---- 13. 评测期统计容器 ----
rep('''    if (steps == 0) {
        // ---------- 评测模式：考卷 + 仿写 ----------''',
    '''    if (steps == 0) {
        // ---------- 评测模式：考卷 + 仿写 ----------
        // ---- 遗忘统计容器（结果写 forget_stats.txt，供 draw_forget_heatmap.py）----
        static double attSum[L][H][T * T];
        static double posLoss[T]; static long long posCnt[T], posHit[T];
        static double byteLoss[256]; static long long byteCnt[256];
        static long long winCnt = 0;
        memset(attSum, 0, sizeof(attSum));''')

# ---- 14. 统计累加（考卷循环里，forward 之后）----
rep('''                forward(g, w.data(), true, false);   // training=false：评测不做 Dropout
                const float* pr = &g.probs[(T - 1) * V];''',
    '''                forward(g, w.data(), true, false);   // training=false：评测不做 Dropout
                winCnt++;
                for (int li = 0; li < L; li++)         // 注意力权重（本 batch 第 0 号窗口）
                    for (int h = 0; h < H; h++) {
                        const float* a = &g.layers[li].att[(size_t)h * T * T];
                        for (int j = 0; j < T * T; j++) attSum[li][h][j] += a[j];
                    }
                for (int t = 0; t < T; t++) {          // 每位置损失/准确率 + 每字节损失
                    int tgt = d[pos + t + 1];
                    const float* pr = &g.probs[(size_t)t * V];
                    posLoss[t] += -logf(max(pr[tgt], 1e-12f)); posCnt[t]++;
                    int best = 0;
                    for (int y = 1; y < V; y++) if (pr[y] > pr[best]) best = y;
                    if (best == tgt) posHit[t]++;
                    byteLoss[tgt] += -logf(max(pr[tgt], 1e-12f)); byteCnt[tgt]++;
                }
                const float* pr = &g.probs[(T - 1) * V];''')

# ---- 15. 三张考卷后写统计文件 ----
rep('''        // 仿写：找"笑道"，用窗口从那里续写 500 字节''',
    '''        // ---- 输出遗忘统计 ----
        {
            ofstream out("forget_stats.txt");
            out << "POS " << T << endl;
            for (int t = 0; t < T; t++)
                out << posLoss[t] / max(1LL, posCnt[t]) << " "
                    << 100.0 * posHit[t] / max(1LL, posCnt[t]) << endl;
            out << "BYTE " << 256 << endl;
            for (int y = 0; y < 256; y++)
                out << byteLoss[y] / max(1LL, byteCnt[y]) << " " << byteCnt[y] << endl;
            out << "ATT " << L << " " << H << " " << T << endl;
            for (int li = 0; li < L; li++)
                for (int h = 0; h < H; h++) {
                    for (int j = 0; j < T * T; j++) out << attSum[li][h][j] / winCnt << " ";
                    out << endl;
                }
            out.close();
        }
        // 仿写：找"笑道"，用窗口从那里续写 500 字节''')

io.open('lesson4_drop.cpp', 'w', encoding='utf-8').write(src)
print('lesson4_drop.cpp written,', len(src), 'chars (was', n0, ')')
