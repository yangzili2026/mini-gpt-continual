// ============================================================
// C++ 造大语言模型 · 第4课：完整迷你 GPT
// ------------------------------------------------------------
// 结构 = 《Attention Is All You Need》右半边（decoder-only）：
//   输入字节 -> 嵌入表E + 位置编码P
//   -> 叠 4 层，每层：
//        层归一化 -> 多头注意力(4头) -> 残差相加
//        层归一化 -> 前馈网络(128->512->128) -> 残差相加
//   -> 层归一化 -> 线性输出(128->256) -> softmax -> 交叉熵
// 训练：完整手写反向传播（和第3课同一套链式法则）
// 语料：all_corpus.bin（179MB 六大风格混合）
// 用法：lesson4_gpt.exe [训练步数] [读取存档] [保存存档]
// ============================================================
#include <bits/stdc++.h>
using namespace std;

// ---------------- 超参数 ----------------
const int V = 256;    // 词表 = 全部字节
const int T = 32;     // 上下文窗口
const int D = 128;    // 向量维度
const int H = 4;      // 注意力头数
const int NH = D / H; // 每个头的维度
const int L = 4;      // 层数
const int F = 512;    // 前馈网络中间层（4×128，与论文一致）
const int B = 16;     // 批量
const int BT = B * T;
const float SQRT_NH = sqrtf((float)NH);

// ---------------- 矩阵工具（行优先） ----------------
void gemm(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[m×k] @ B[k×n]
    memset(C, 0, sizeof(float) * (size_t)m * n);
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++) {
            float a = A[i * k + l];
            const float* Brow = Bm + (size_t)l * n;
            float* Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
}
void gemm_bt(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[m×k] @ B[n×k]^T
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[j * k + l];
            C[i * n + j] = s;
        }
}
void gemm_at(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[k×m]^T @ B[k×n]
    memset(C, 0, sizeof(float) * (size_t)m * n);
    for (int l = 0; l < k; l++)
        for (int i = 0; i < m; i++) {
            float a = A[(size_t)l * m + i];
            const float* Brow = Bm + (size_t)l * n;
            float* Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
}
void bias_add(float* X, const float* bias, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) X[i * cols + j] += bias[j];
}
void colsum(const float* X, float* bias, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        float s = 0;
        for (int i = 0; i < rows; i++) s += X[i * cols + j];
        bias[j] += s;
    }
}

// ---------------- 层归一化 ----------------
// 【难】LayerNorm：每行(D个数)变成均值0方差1，再缩放平移。
// 它是第3课 NaN 爆炸的解药：不管数字滚多大，LN 都会把它们拉回标准尺度。
void layernorm_fwd(const float* x, const float* g, const float* b,
                   float* xhat, float* rstd, float* y, int rows) {
    for (int i = 0; i < rows; i++) {
        const float* xr = x + (size_t)i * D;
        float mean = 0, var = 0;
        for (int j = 0; j < D; j++) mean += xr[j];
        mean /= D;
        for (int j = 0; j < D; j++) var += (xr[j] - mean) * (xr[j] - mean);
        float rs = 1.0f / sqrtf(var / D + 1e-5f);
        rstd[i] = rs;
        for (int j = 0; j < D; j++) {
            xhat[i * D + j] = (xr[j] - mean) * rs;
            y[i * D + j] = g[j] * xhat[i * D + j] + b[j];
        }
    }
}
void layernorm_bwd(const float* xhat, const float* rstd, const float* g,
                   const float* dy, float* dx, float* dg, float* db, int rows) {
    for (int i = 0; i < rows; i++) {
        const float* hr = xhat + (size_t)i * D;
        const float* dyr = dy + (size_t)i * D;
        float rs = rstd[i];
        float dsum = 0, ddot = 0;
        for (int j = 0; j < D; j++) {
            dg[j] += dyr[j] * hr[j];
            db[j] += dyr[j];
            dsum += dyr[j] * g[j];
            ddot += dyr[j] * g[j] * hr[j];
        }
        dsum /= D; ddot /= D;
        // 【难】LN 反传公式：dx = rstd * (g*dy - 均值(g*dy) - xhat*均值(g*dy*xhat))
        for (int j = 0; j < D; j++)
            dx[i * D + j] = rs * (dyr[j] * g[j] - dsum - hr[j] * ddot);
    }
}

// ============================================================
// 模型
// ============================================================
struct Layer {
    // 参数
    vector<float> ln1g = vector<float>(D), ln1b = vector<float>(D),
                  Wq = vector<float>(D * D), bq = vector<float>(D),
                  Wk = vector<float>(D * D), bk = vector<float>(D),
                  Wv = vector<float>(D * D), bv = vector<float>(D),
                  Wo = vector<float>(D * D), bo = vector<float>(D),
                  ln2g = vector<float>(D), ln2b = vector<float>(D),
                  W1 = vector<float>(D * F), b1 = vector<float>(F),
                  W2 = vector<float>(F * D), b2 = vector<float>(D);
    // 梯度
    vector<float> g_ln1g = vector<float>(D), g_ln1b = vector<float>(D),
                  g_Wq = vector<float>(D * D), g_bq = vector<float>(D),
                  g_Wk = vector<float>(D * D), g_bk = vector<float>(D),
                  g_Wv = vector<float>(D * D), g_bv = vector<float>(D),
                  g_Wo = vector<float>(D * D), g_bo = vector<float>(D),
                  g_ln2g = vector<float>(D), g_ln2b = vector<float>(D),
                  g_W1 = vector<float>(D * F), g_b1 = vector<float>(F),
                  g_W2 = vector<float>(F * D), g_b2 = vector<float>(D);
    // 前向缓存（反向传播要重用）
    vector<float> x_in, xhat1, rstd1, y1, q, k, v, att, ctx,
                  x_in2, xhat2, rstd2, y2, hid, ffn;
    Layer() {
        x_in.resize(BT * D); xhat1.resize(BT * D); rstd1.resize(BT);
        y1.resize(BT * D);
        q.resize(BT * D); k.resize(BT * D); v.resize(BT * D);
        att.resize((size_t)B * H * T * T); ctx.resize(BT * D);
        x_in2.resize(BT * D); xhat2.resize(BT * D); rstd2.resize(BT);
        y2.resize(BT * D);
        hid.resize((size_t)BT * F); ffn.resize(BT * D);
    }
    void zeroGrad() {
        for (auto p : {&g_ln1g, &g_ln1b, &g_Wq, &g_bq, &g_Wk, &g_bk, &g_Wv, &g_bv,
                       &g_Wo, &g_bo, &g_ln2g, &g_ln2b, &g_W1, &g_b1, &g_W2, &g_b2})
            fill(p->begin(), p->end(), 0.0f);
    }
};

struct GPT {
    vector<float> E = vector<float>(V * D), P = vector<float>(T * D),
                  lnFg = vector<float>(D), lnFb = vector<float>(D),
                  WU = vector<float>(D * V);
    vector<float> g_E = vector<float>(V * D), g_P = vector<float>(T * D),
                  g_lnFg = vector<float>(D), g_lnFb = vector<float>(D),
                  g_WU = vector<float>(D * V);
    vector<float> x, xhatf, rstdf, yf, probs;
    vector<Layer> layers{L};
    GPT() {
        x.resize(BT * D); xhatf.resize(BT * D); rstdf.resize(BT);
        yf.resize(BT * D); probs.resize((size_t)BT * V);
    }
    long long nparams() {
        long long n = E.size() + P.size() + lnFg.size() + lnFb.size() + WU.size();
        for (auto& l : layers) n += l.ln1g.size() * 2 + l.Wq.size() * 4 + l.bq.size() * 4
            + l.ln2g.size() * 2 + l.W1.size() + l.b1.size() + l.W2.size() + l.b2.size();
        return n;
    }
};

// ---------------- 初始化 ----------------
void initGPT(GPT& g, mt19937& rng) {
    normal_distribution<float> nd(0.0f, 0.02f);
    auto filln = [&](vector<float>& v) {
        for (auto& x : v) x = nd(rng);
    };
    filln(g.E); filln(g.P); filln(g.WU);
    for (auto& l : g.layers) {
        filln(l.Wq); filln(l.Wk); filln(l.Wv); filln(l.Wo);
        filln(l.W1); filln(l.W2);
        fill(l.ln1g.begin(), l.ln1g.end(), 1.0f);   // 【记】LN 缩放初始化为1
        fill(l.ln2g.begin(), l.ln2g.end(), 1.0f);
        fill(l.ln1b.begin(), l.ln1b.end(), 0.0f); fill(l.ln2b.begin(), l.ln2b.end(), 0.0f);
        fill(l.bq.begin(), l.bq.end(), 0.0f); fill(l.bk.begin(), l.bk.end(), 0.0f);
        fill(l.bv.begin(), l.bv.end(), 0.0f); fill(l.bo.begin(), l.bo.end(), 0.0f);
        fill(l.b1.begin(), l.b1.end(), 0.0f); fill(l.b2.begin(), l.b2.end(), 0.0f);
    }
    fill(g.lnFg.begin(), g.lnFg.end(), 1.0f);
    fill(g.lnFb.begin(), g.lnFb.end(), 0.0f);
}

// ============================================================
// 前向：toks 是 B×T 个字节编号，返回平均损失
// ============================================================
float forward(GPT& g, const unsigned char* toks, bool needLast = false) {
    // 1. 嵌入 + 位置编码
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const float* er = &g.E[toks[b * (T + 1) + t] * D];
            const float* pr = &g.P[(size_t)t * D];
            float* xr = &g.x[(b * T + t) * D];
            for (int d = 0; d < D; d++) xr[d] = er[d] + pr[d];
        }

    // 2. 每一层
    static vector<float> attout;
    if (attout.empty()) attout.resize(BT * D);
    for (auto& l : g.layers) {
        copy(g.x.begin(), g.x.end(), l.x_in.begin());
        layernorm_fwd(g.x.data(), l.ln1g.data(), l.ln1b.data(),
                      l.xhat1.data(), l.rstd1.data(), l.y1.data(), BT);
        gemm(l.y1.data(), l.Wq.data(), l.q.data(), BT, D, D); bias_add(l.q.data(), l.bq.data(), BT, D);
        gemm(l.y1.data(), l.Wk.data(), l.k.data(), BT, D, D); bias_add(l.k.data(), l.bk.data(), BT, D);
        gemm(l.y1.data(), l.Wv.data(), l.v.data(), BT, D, D); bias_add(l.v.data(), l.bv.data(), BT, D);

        // 多头注意力（逐 batch、逐头，逻辑最清楚）
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                float* att = &l.att[((size_t)b * H + h) * T * T];
                float* qh = &l.q[(size_t)b * T * D + h * NH];
                float* kh = &l.k[(size_t)b * T * D + h * NH];
                float* vh = &l.v[(size_t)b * T * D + h * NH];
                float* ch = &l.ctx[(size_t)b * T * D + h * NH];
                for (int t = 0; t < T; t++) {
                    float mx = -1e30f, ssum = 0;
                    for (int j = 0; j <= t; j++) {
                        float s = 0;
                        for (int i = 0; i < NH; i++) s += qh[t * D + i] * kh[j * D + i];
                        att[t * T + j] = s / SQRT_NH;
                        mx = max(mx, att[t * T + j]);
                    }
                    for (int j = 0; j <= t; j++) { att[t * T + j] = expf(att[t * T + j] - mx); ssum += att[t * T + j]; }
                    for (int j = 0; j <= t; j++) att[t * T + j] /= ssum;
                    for (int j = t + 1; j < T; j++) att[t * T + j] = 0;   // 【易错】掩码补0
                }
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < NH; i++) {
                        float s = 0;
                        for (int j = 0; j <= t; j++) s += att[t * T + j] * vh[j * D + i];
                        ch[t * D + i] = s;
                    }
            }

        // 输出投影 + 残差
        gemm(l.ctx.data(), l.Wo.data(), attout.data(), BT, D, D);
        bias_add(attout.data(), l.bo.data(), BT, D);
        for (int i = 0; i < BT * D; i++) g.x[i] += attout[i];

        // 前馈网络 + 残差
        copy(g.x.begin(), g.x.end(), l.x_in2.begin());
        layernorm_fwd(g.x.data(), l.ln2g.data(), l.ln2b.data(),
                      l.xhat2.data(), l.rstd2.data(), l.y2.data(), BT);
        gemm(l.y2.data(), l.W1.data(), l.hid.data(), BT, D, F);
        bias_add(l.hid.data(), l.b1.data(), BT, F);
        for (auto& h_ : l.hid) h_ = max(h_, 0.0f);      // ReLU
        gemm(l.hid.data(), l.W2.data(), l.ffn.data(), BT, F, D);
        bias_add(l.ffn.data(), l.b2.data(), BT, D);
        for (int i = 0; i < BT * D; i++) g.x[i] += l.ffn[i];
    }

    // 3. 顶层 LN + 输出头 + softmax
    layernorm_fwd(g.x.data(), g.lnFg.data(), g.lnFb.data(),
                  g.xhatf.data(), g.rstdf.data(), g.yf.data(), BT);
    gemm(g.yf.data(), g.WU.data(), g.probs.data(), BT, D, V);  // 先存 logits
    int tlim = T;   // 【易错】最后一列也监督，必须 softmax，不能省
    for (int b = 0; b < B; b++)
        for (int t = 0; t < tlim; t++) {
            float* row = &g.probs[((size_t)b * T + t) * V];
            float mx = -1e30f, s = 0;
            for (int y = 0; y < V; y++) mx = max(mx, row[y]);
            for (int y = 0; y < V; y++) { row[y] = expf(row[y] - mx); s += row[y]; }
            for (int y = 0; y < V; y++) row[y] /= s;
        }

    float loss = 0; int cnt = 0;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int target = toks[b * (T + 1) + t + 1];
            loss += -logf(max(g.probs[((size_t)b * T + t) * V + target], 1e-12f));
            cnt++;
        }
    return loss / cnt;
}

// ============================================================
// 反向传播
// ============================================================
void backward(GPT& g, const unsigned char* toks) {
    static vector<float> dx, dy, tmp, dq, dk, dv, dctx2, datt, dffn, dlogits;
    if (dx.empty()) {
        dx.resize(BT * D); dy.resize(BT * D); tmp.resize(BT * F);   // 【易错】tmp 要装 FFN 的 BT×F
        dq.resize(BT * D); dk.resize(BT * D); dv.resize(BT * D); dctx2.resize(BT * D);
        datt.resize(T * T); dffn.resize(BT * D);
        dlogits.resize((size_t)BT * V);
    }
    const int NVALID = B * T;
    fill(dx.begin(), dx.end(), 0.0f);
    fill(dlogits.begin(), dlogits.end(), 0.0f);

    // 1. softmax+交叉熵 的梯度 = (概率 - 正确答案) / 样本数
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            float* row = &dlogits[((size_t)b * T + t) * V];
            const float* pr = &g.probs[((size_t)b * T + t) * V];
            int target = toks[b * (T + 1) + t + 1];
            for (int y = 0; y < V; y++)
                row[y] = (pr[y] - (y == target ? 1.0f : 0.0f)) / NVALID;
        }

    // 2. 输出头：dWU = yf^T @ dlogits；dx(顶层LN输出) = dlogits @ WU^T
    gemm_at(g.yf.data(), dlogits.data(), g.g_WU.data(), D, BT, V);
    gemm_bt(dlogits.data(), g.WU.data(), dx.data(), BT, V, D);

    // 3. 顶层 LayerNorm 反传（dx 即 d(yf)，传出 d(x) 写进 dy，再交还给 dx）
    layernorm_bwd(g.xhatf.data(), g.rstdf.data(), g.lnFg.data(), dx.data(), dy.data(),
                  g.g_lnFg.data(), g.g_lnFb.data(), BT);
    copy(dy.begin(), dy.end(), dx.begin());

    // 4. 逐层反传（倒序）
    for (int li = L - 1; li >= 0; li--) {
        auto& l = g.layers[li];

        // --- 前馈网络支路 ---
        copy(dx.begin(), dx.end(), dffn.begin());           // dffn = dx
        colsum(dffn.data(), l.g_b2.data(), BT, D);
        gemm_at(l.hid.data(), dffn.data(), l.g_W2.data(), F, BT, D);
        gemm_bt(dffn.data(), l.W2.data(), tmp.data(), BT, D, F);   // dhid（ReLU前）
        for (int i = 0; i < BT * F; i++) if (l.hid[i] == 0) tmp[i] = 0;   // ReLU反传
        colsum(tmp.data(), l.g_b1.data(), BT, F);
        gemm_at(l.y2.data(), tmp.data(), l.g_W1.data(), D, BT, F);
        gemm_bt(tmp.data(), l.W1.data(), dy.data(), BT, F, D);     // d(y2)
        layernorm_bwd(l.xhat2.data(), l.rstd2.data(), l.ln2g.data(), dy.data(),
                      tmp.data(), l.g_ln2g.data(), l.g_ln2b.data(), BT);
        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];   // 残差：dx 变为 d(x_in2)

        // --- 注意力支路 ---
        colsum(dx.data(), l.g_bo.data(), BT, D);                    // dbo
        gemm_at(l.ctx.data(), dx.data(), l.g_Wo.data(), D, BT, D);  // dWo
        gemm_bt(dx.data(), l.Wo.data(), dctx2.data(), BT, D, D);    // d(ctx)

        fill(dq.begin(), dq.end(), 0.0f);
        fill(dk.begin(), dk.end(), 0.0f);
        fill(dv.begin(), dv.end(), 0.0f);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                const float* att = &l.att[((size_t)b * H + h) * T * T];
                const float* qh = &l.q[(size_t)b * T * D + h * NH];
                const float* kh = &l.k[(size_t)b * T * D + h * NH];
                const float* vh = &l.v[(size_t)b * T * D + h * NH];
                float* cq = &dq[(size_t)b * T * D + h * NH];
                float* ck = &dk[(size_t)b * T * D + h * NH];
                float* cv = &dv[(size_t)b * T * D + h * NH];
                // dctx[t] 切片
                auto dcx = [&](int t, int i) {
                    return dctx2[((size_t)b * T + t) * D + h * NH + i];
                };
                // dv[j] = Σ_{t≥j} att[t,j] · dctx[t]
                for (int j = 0; j < T; j++)
                    for (int i = 0; i < NH; i++) {
                        float s = 0;
                        for (int t = j; t < T; t++) s += att[t * T + j] * dcx(t, i);
                        cv[j * D + i] = s;
                    }
                // dscore（softmax 反传）
                fill(datt.begin(), datt.end(), 0.0f);
                for (int t = 0; t < T; t++) {
                    float rowsum = 0;
                    for (int j = 0; j <= t; j++) {
                        float s = 0;
                        for (int i = 0; i < NH; i++) s += dcx(t, i) * vh[j * D + i];
                        datt[t * T + j] = s;
                        rowsum += att[t * T + j] * s;
                    }
                    for (int j = 0; j <= t; j++)
                        datt[t * T + j] = att[t * T + j] * (datt[t * T + j] - rowsum) / SQRT_NH;
                }
                // dq[t] = Σ_{j≤t} dscore[t,j] · k[j] ；dk[j] = Σ_{t≥j} dscore[t,j] · q[t]
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < NH; i++) {
                        float sq = 0;
                        for (int j = 0; j <= t; j++) sq += datt[t * T + j] * kh[j * D + i];
                        cq[t * D + i] = sq;
                    }
                for (int j = 0; j < T; j++)
                    for (int i = 0; i < NH; i++) {
                        float sk = 0;
                        for (int t = j; t < T; t++) sk += datt[t * T + j] * qh[t * D + i];
                        ck[j * D + i] = sk;
                    }
            }
        // 拼头反传到 y1：dy = dq@Wq^T + dk@Wk^T + dv@Wv^T
        gemm_bt(dq.data(), l.Wq.data(), dy.data(), BT, D, D);
        gemm_bt(dk.data(), l.Wk.data(), tmp.data(), BT, D, D);
        for (int i = 0; i < BT * D; i++) dy[i] += tmp[i];
        gemm_bt(dv.data(), l.Wv.data(), tmp.data(), BT, D, D);
        for (int i = 0; i < BT * D; i++) dy[i] += tmp[i];
        // 三个矩阵的梯度
        gemm_at(l.y1.data(), dq.data(), l.g_Wq.data(), D, BT, D); colsum(dq.data(), l.g_bq.data(), BT, D);
        gemm_at(l.y1.data(), dk.data(), l.g_Wk.data(), D, BT, D); colsum(dk.data(), l.g_bk.data(), BT, D);
        gemm_at(l.y1.data(), dv.data(), l.g_Wv.data(), D, BT, D); colsum(dv.data(), l.g_bv.data(), BT, D);
        // LN1 反传 + 残差
        layernorm_bwd(l.xhat1.data(), l.rstd1.data(), l.ln1g.data(), dy.data(),
                      tmp.data(), l.g_ln1g.data(), l.g_ln1b.data(), BT);
        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];
    }

    // 5. 嵌入和位置编码的梯度
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const float* xr = &dx[((size_t)b * T + t) * D];
            float* er = &g.g_E[toks[b * (T + 1) + t] * D];
            float* pr = &g.g_P[(size_t)t * D];
            for (int d = 0; d < D; d++) { er[d] += xr[d]; pr[d] += xr[d]; }
        }
}

// ---------------- Adam ----------------
struct Adam {
    unordered_map<vector<float>*, vector<float>> m1, m2;
    void step(vector<float>& p, vector<float>& g, int t, float lr) {
        auto& M = m1[&p]; auto& Vv = m2[&p];
        if (M.empty()) { M.assign(p.size(), 0); Vv.assign(p.size(), 0); }
        float c1 = 1 - powf(0.9f, t), c2 = 1 - powf(0.999f, t);
        for (size_t i = 0; i < p.size(); i++) {
            M[i] = 0.9f * M[i] + 0.1f * g[i];
            Vv[i] = 0.999f * Vv[i] + 0.001f * g[i] * g[i];
            p[i] -= lr * (M[i] / c1) / (sqrtf(Vv[i] / c2) + 1e-8f);
        }
    }
};
void applyAdam(GPT& g, Adam& a, int t, float lr) {
    a.step(g.E, g.g_E, t, lr); a.step(g.P, g.g_P, t, lr);
    a.step(g.lnFg, g.g_lnFg, t, lr); a.step(g.lnFb, g.g_lnFb, t, lr);
    a.step(g.WU, g.g_WU, t, lr);
    for (auto& l : g.layers) {
        a.step(l.ln1g, l.g_ln1g, t, lr); a.step(l.ln1b, l.g_ln1b, t, lr);
        a.step(l.Wq, l.g_Wq, t, lr); a.step(l.bq, l.g_bq, t, lr);
        a.step(l.Wk, l.g_Wk, t, lr); a.step(l.bk, l.g_bk, t, lr);
        a.step(l.Wv, l.g_Wv, t, lr); a.step(l.bv, l.g_bv, t, lr);
        a.step(l.Wo, l.g_Wo, t, lr); a.step(l.bo, l.g_bo, t, lr);
        a.step(l.ln2g, l.g_ln2g, t, lr); a.step(l.ln2b, l.g_ln2b, t, lr);
        a.step(l.W1, l.g_W1, t, lr); a.step(l.b1, l.g_b1, t, lr);
        a.step(l.W2, l.g_W2, t, lr); a.step(l.b2, l.g_b2, t, lr);
    }
}

// ============================================================
// 主流程
// ============================================================
int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 100;
    const char* loadfile = argc > 2 ? argv[2] : nullptr;
    const char* savefile = argc > 3 ? argv[3] : "gpt4.bin";

    ifstream fin("all_corpus_cn.bin", ios::binary);
    vector<unsigned char> data((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();
    if (data.size() < 1000000) {   // 【易错】文件打不开时给个明白话，不要神秘崩溃
        cout << "打不开 all_corpus_cn.bin！请在 cpp 目录下运行" << endl;
        return 1;
    }
    cout << "语料: all_corpus.bin " << data.size() / 1000000 << " MB" << endl;

    mt19937 rng(42);
    GPT g;
    initGPT(g, rng);
    cout << "参数量: " << g.nparams() / 1000 << " K" << endl;

    if (loadfile) {
        ifstream lin(loadfile, ios::binary);
        if (lin) {
            auto rd = [&](vector<float>& v) { lin.read((char*)v.data(), v.size() * 4); };
            rd(g.E); rd(g.P); rd(g.lnFg); rd(g.lnFb); rd(g.WU);
            for (auto& l : g.layers) {
                rd(l.ln1g); rd(l.ln1b); rd(l.Wq); rd(l.bq); rd(l.Wk); rd(l.bk);
                rd(l.Wv); rd(l.bv); rd(l.Wo); rd(l.bo);
                rd(l.ln2g); rd(l.ln2b); rd(l.W1); rd(l.b1); rd(l.W2); rd(l.b2);
            }
            cout << "已读取存档 " << loadfile << endl;
        }
    }

    if (steps == 0) {
        // ---------- 评测模式：考卷 + 仿写 ----------
        auto exam = [&](const char* name, const unsigned char* d, size_t n) {
            double loss = 0; long long hit = 0, cnt = 0;
            for (size_t pos = 0; pos + T < n && cnt < 3000; pos++) {
                static vector<unsigned char> w;
                if (w.empty()) w.resize(B * (T + 1));
                for (int t = 0; t < T; t++) w[t] = d[pos + t];
                forward(g, w.data(), true);
                const float* pr = &g.probs[(T - 1) * V];
                int target = d[pos + T];
                loss += -logf(max(pr[target], 1e-12f));
                int best = 0;
                for (int y = 1; y < V; y++) if (pr[y] > pr[best]) best = y;
                if (best == target) hit++;
                cnt++;
            }
            printf("考卷[%s] 平均损失 %.3f  准确率 %.1f%%\n", name, loss / cnt, 100.0 * hit / cnt);
        };
        // 考卷：红楼梦原文 + 三体 + 天工切片（每个只考前 6 万字节）
        ifstream f1("hongloumeng.txt", ios::binary);
        vector<unsigned char> d1((istreambuf_iterator<char>(f1)), istreambuf_iterator<char>());
        exam("红楼梦·古文", d1.data(), 60000);
        ifstream f2("santi1.txt", ios::binary);
        vector<unsigned char> d2((istreambuf_iterator<char>(f2)), istreambuf_iterator<char>());
        exam("三体·科幻", d2.data(), 60000);
        ifstream f3("zh_modern.txt", ios::binary);
        vector<unsigned char> d3((istreambuf_iterator<char>(f3)), istreambuf_iterator<char>());
        exam("天工·现代文", d3.data(), 60000);

        // 仿写：找"笑道"，用窗口从那里续写 500 字节
        int found = -1;
        for (size_t i = 8; i + 6 < data.size(); i++)
            if (data[i]==231 && data[i+1]==172 && data[i+2]==145 &&
                data[i+3]==233 && data[i+4]==129 && data[i+5]==147) { found = i; break; }
        vector<unsigned char> w(B * (T + 1));
        for (int t = 0; t < T; t++) w[t] = data[found - (T - 6) + t];
        string out;
        for (int gstep = 0; gstep < 500; gstep++) {
            forward(g, w.data(), true);
            const float* pr = &g.probs[(T - 1) * V];
            double q[256], s = 0;
            for (int y = 0; y < 256; y++) { q[y] = pow(pr[y], 1.0 / 0.7); s += q[y]; }
            double r = uniform_real_distribution<double>(0, 1)(rng) * s;
            int nxt = 0;
            while (r > q[nxt]) { r -= q[nxt]; nxt++; }
            out.push_back((char)w[0]);
            for (int t = 0; t < T - 1; t++) w[t] = w[t + 1];
            w[T - 1] = (unsigned char)nxt;
        }
        out.append((char*)&w[0], T);
        ofstream fout("gen4.bin", ios::binary);
        fout.write(out.data(), out.size());
        fout.close();
        cout << "仿写已存 gen4.bin（500字节）" << endl;
        return 0;
    }

    Adam adam;
    vector<unsigned char> batch(B * (T + 1));
    double runLoss = 0; int runN = 0;
    time_t t0 = time(nullptr);
    for (int step = 1; step <= steps; step++) {
        for (int b = 0; b < B; b++) {
            size_t pos = rng() % (data.size() - T);
            for (int t = 0; t <= T; t++) batch[b * (T + 1) + t] = data[pos + t];
        }
        for (auto& l : g.layers) l.zeroGrad();
        fill(g.g_E.begin(), g.g_E.end(), 0.0f);
        fill(g.g_P.begin(), g.g_P.end(), 0.0f);
        fill(g.g_lnFg.begin(), g.g_lnFg.end(), 0.0f);
        fill(g.g_lnFb.begin(), g.g_lnFb.end(), 0.0f);
        fill(g.g_WU.begin(), g.g_WU.end(), 0.0f);

        float loss = forward(g, batch.data());
        backward(g, batch.data());
        float lr = step < steps * 0.6 ? 0.001f : 0.0003f;   // 先大后小
        applyAdam(g, adam, step, lr);

        runLoss += loss; runN++;
        if (step % 20 == 0) {
            cout << "step " << step << "  平均损失 = " << fixed << setprecision(3)
                 << runLoss / runN << "  (随机 5.545)  每20步用时 "
                 << time(nullptr) - t0 << "s" << endl;
            runLoss = 0; runN = 0; t0 = time(nullptr);
        }
        if (step % 100 == 0) {   // 【记】每100步自动存档，超时/断电也不丢进度
            ofstream lout(savefile, ios::binary);
            auto wr = [&](vector<float>& v) { lout.write((char*)v.data(), v.size() * 4); };
            wr(g.E); wr(g.P); wr(g.lnFg); wr(g.lnFb); wr(g.WU);
            for (auto& l : g.layers) {
                wr(l.ln1g); wr(l.ln1b); wr(l.Wq); wr(l.bq); wr(l.Wk); wr(l.bk);
                wr(l.Wv); wr(l.bv); wr(l.Wo); wr(l.bo);
                wr(l.ln2g); wr(l.ln2b); wr(l.W1); wr(l.b1); wr(l.W2); wr(l.b2);
            }
            lout.close();
        }
    }
    cout << "存档已存 " << savefile << endl;
    return 0;
}
