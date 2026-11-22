// ============================================================
// C++ 造大语言模型 · 第5课续：持续学习主实验台（论文第3章全套实验）
// ------------------------------------------------------------
// 结构 = 《Attention Is All You Need》右半边（decoder-only）：
//   输入字节 -> 嵌入表E + 位置编码P
//   -> 叠 4 层，每层：
//        层归一化 -> 多头注意力(4头) -> 残差相加
//        层归一化 -> 前馈网络(128->512->128) -> 残差相加
//   -> 层归一化 -> 线性输出(128->256) -> softmax -> 交叉熵
// 训练：完整手写反向传播（和第3课同一套链式法则）
// 复用第6课全部模型代码（前向/反向/Adam/GEMM），只换主流程
// ============================================================
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <string>
#include <ctime>
#include <unordered_map>
#include <functional>
#include <unordered_set>
#include <cstdint>
#include <cuda_runtime.h>
using namespace std;

// ---------------- 超参数 ----------------
const int V = 256;    // 词表 = 全部字节
const int T = 128;    // 上下文窗口（第6课：32 -> 128，翻两番）
const int D = 128;    // 向量维度
const int H = 4;      // 注意力头数
const int NH = D / H; // 每个头的维度
const int L = 4;      // 层数
const int F = 512;    // 前馈网络中间层（4×128，与论文一致）
const int B = 16;     // 批量
// 【记】G0 预训练总步数：分段续训时学习率按这个总目标退火（6成步数后 0.001 -> 0.0003）
const int TOTAL_STEPS = 12000;
const int BT = B * T;
const float SQRT_NH = sqrtf((float)NH);

// ---------------- 矩阵工具（行优先） ----------------
void gemm_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {
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
void gemm_bt_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[m×k] @ B[n×k]^T
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[j * k + l];
            C[i * n + j] = s;
        }
}
void gemm_at_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {
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

// ============================================================
// 第5课：GPU 矩阵乘（kernel = 在显卡上跑的函数）
// 三个 kernel 和上面三个 CPU 版一一对应，格子算法相同，只是格子分发给线程
// ============================================================
__global__ void gemm_k(const float* A, const float* Bm, float* C, int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float s = 0;
    for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[l * n + j];
    C[i * n + j] = s;
}
__global__ void gemm_bt_k(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[m×k] @ B[n×k]^T（Bm 按行存，取 Bm[j*k+l]）
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float s = 0;
    for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[j * k + l];
    C[i * n + j] = s;
}
__global__ void gemm_at_k(const float* A, const float* Bm, float* C, int m, int k, int n) {
    // C[m×n] = A[k×m]^T @ B[k×n]（A 按 k 行 m 列存，取 A[l*m+i]）
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float s = 0;
    for (int l = 0; l < k; l++) s += A[l * m + i] * Bm[l * n + j];
    C[i * n + j] = s;
}

// ---- 显卡上的缓冲区（只申请一次，反复复用） ----
float *g_dA = nullptr, *g_dB = nullptr, *g_dC = nullptr;
const size_t GMAX = 2048 * 512;             // 最大矩阵：FFN 隐藏层 2048×512
bool USE_GPU = true;                        // 命令行第4参数 cpu 可关掉，方便对照

// 三个分发函数：和 CPU 版同名同参数，哪里都不用改，自动换显卡算
void gemm(const float* A, const float* Bm, float* C, int m, int k, int n) {
    if (!USE_GPU) { gemm_cpu(A, Bm, C, m, k, n); return; }
    if (!g_dA) { cudaMalloc(&g_dA, GMAX * 4); cudaMalloc(&g_dB, GMAX * 4); cudaMalloc(&g_dC, GMAX * 4); }
    cudaMemcpy(g_dA, A, (size_t)m * k * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dB, Bm, (size_t)k * n * 4, cudaMemcpyHostToDevice);
    dim3 blk(16, 16), grd((n + 15) / 16, (m + 15) / 16);
    gemm_k<<<grd, blk>>>(g_dA, g_dB, g_dC, m, k, n);
    cudaMemcpy(C, g_dC, (size_t)m * n * 4, cudaMemcpyDeviceToHost);
}
void gemm_bt(const float* A, const float* Bm, float* C, int m, int k, int n) {
    if (!USE_GPU) { gemm_bt_cpu(A, Bm, C, m, k, n); return; }
    if (!g_dA) { cudaMalloc(&g_dA, GMAX * 4); cudaMalloc(&g_dB, GMAX * 4); cudaMalloc(&g_dC, GMAX * 4); }
    cudaMemcpy(g_dA, A, (size_t)m * k * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dB, Bm, (size_t)n * k * 4, cudaMemcpyHostToDevice);
    dim3 blk(16, 16), grd((n + 15) / 16, (m + 15) / 16);
    gemm_bt_k<<<grd, blk>>>(g_dA, g_dB, g_dC, m, k, n);
    cudaMemcpy(C, g_dC, (size_t)m * n * 4, cudaMemcpyDeviceToHost);
}
void gemm_at(const float* A, const float* Bm, float* C, int m, int k, int n) {
    if (!USE_GPU) { gemm_at_cpu(A, Bm, C, m, k, n); return; }
    if (!g_dA) { cudaMalloc(&g_dA, GMAX * 4); cudaMalloc(&g_dB, GMAX * 4); cudaMalloc(&g_dC, GMAX * 4); }
    cudaMemcpy(g_dA, A, (size_t)k * m * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dB, Bm, (size_t)k * n * 4, cudaMemcpyHostToDevice);
    dim3 blk(16, 16), grd((n + 15) / 16, (m + 15) / 16);
    gemm_at_k<<<grd, blk>>>(g_dA, g_dB, g_dC, m, k, n);
    cudaMemcpy(C, g_dC, (size_t)m * n * 4, cudaMemcpyDeviceToHost);
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
        dx.resize(BT * D); dy.resize(BT * D); tmp.resize(BT * F);   // 【易错】tmp 要装 FFN 的 BT×F，写 BT×D 会越界崩堆
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
// 主流程：持续学习实验台（论文 3.1-3.4 全套）
// 用法：
//   lesson5_continual.exe t1 <G0存档> <步数>          训 T1 一次，存共享分叉点 t1_fork.bin
//   lesson5_continual.exe arm <组> <种子> <每域步数> [resume]  从分叉点跑 T2->T3
//   lesson5_continual.exe gen <存档> <提示源> <输出> <字节数> <种子> <温度>  自采样（坍缩用）
//   lesson5_continual.exe trainfile <存档> <语料> <步数> <新存档> <种子>     任意语料续训（坍缩用）
//   lesson5_continual.exe evalfile <存档> <语料>     打印语料平均损失（自损失用）
// 组：A 不复习 / B0 未过滤自回放 / B1 UTF-8过滤 / B2 自困惑度过滤 / B3 n-gram过滤 / C 真实回放 / D 手术
// ============================================================

vector<unsigned char> readbin(const char* path) {   // 【易错】整块读，别用迭代器（慢10倍）
    FILE* f = fopen(path, "rb");
    if (!f) { printf("打不开 %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    vector<unsigned char> d(n);
    size_t r = fread(d.data(), 1, n, f);
    fclose(f);
    if (r != (size_t)n) { printf("没读完整 %s\n", path); exit(1); }
    return d;
}
void saveckpt(const char* path, GPT& g) {
    ofstream out(path, ios::binary);
    auto wr = [&](vector<float>& v) { out.write((char*)v.data(), v.size() * 4); };
    wr(g.E); wr(g.P); wr(g.lnFg); wr(g.lnFb); wr(g.WU);
    for (auto& l : g.layers) {
        wr(l.ln1g); wr(l.ln1b); wr(l.Wq); wr(l.bq); wr(l.Wk); wr(l.bk);
        wr(l.Wv); wr(l.bv); wr(l.Wo); wr(l.bo);
        wr(l.ln2g); wr(l.ln2b); wr(l.W1); wr(l.b1); wr(l.W2); wr(l.b2);
    }
    out.close();
}
void loadckpt(const char* path, GPT& g) {
    ifstream in(path, ios::binary);
    if (!in) { printf("打不开存档 %s\n", path); exit(1); }
    auto rd = [&](vector<float>& v) { in.read((char*)v.data(), v.size() * 4); };
    rd(g.E); rd(g.P); rd(g.lnFg); rd(g.lnFb); rd(g.WU);
    for (auto& l : g.layers) {
        rd(l.ln1g); rd(l.ln1b); rd(l.Wq); rd(l.bq); rd(l.Wk); rd(l.bk);
        rd(l.Wv); rd(l.bv); rd(l.Wo); rd(l.bo);
        rd(l.ln2g); rd(l.ln2b); rd(l.W1); rd(l.b1); rd(l.W2); rd(l.b2);
    }
    in.close();
    printf("已读取存档 %s\n", path);
}

// ---------------- 免费信号 ----------------
// 【记】UTF-8 结构合法性：按编码规则解码，返回合法字符占比（成本为零，论文核心信号）
float utf8_legality(const unsigned char* s, int n) {
    int chars = 0, valid = 0, i = 0;
    while (i < n) {
        unsigned char c = s[i];
        int len = 0;
        if (c < 0x80) { len = 1; }
        else if ((c & 0xE0) == 0xC0) { len = 2; if (c < 0xC2) len = 0; }        // 【易错】0xC0/0xC1 是非法前导
        else if ((c & 0xF0) == 0xE0) { len = 3; if (c == 0xE0 && i+1 < n && s[i+1] < 0xA0) len = 0; }
        else if ((c & 0xF8) == 0xF0) { len = 4; if (c == 0xF0 && i+1 < n && s[i+1] < 0x90) len = 0;
                                         if (c > 0xF4) len = 0; }
        else { len = 0; }
        bool ok = len > 0;
        for (int k = 1; ok && k < len; k++) {
            if (i + k >= n || (s[i + k] & 0xC0) != 0x80) ok = false;   // 后续字节必须是 10xxxxxx
        }
        if (ok) { valid++; i += len; } else { i++; }   // 【记】非法字节按1步跳过，别死循环
        chars++;
    }
    return chars ? (float)valid / chars : 0.0f;
}

// 【记】n-gram 重合度：候选的 3-gram(字节) 有多少命中种子银行（成本为零）
float ngram_overlap(const unsigned char* s, int n, const unordered_set<uint32_t>& seed_grams) {
    if (n < 3) return 0.0f;
    unordered_set<uint32_t> cand;
    for (int i = 0; i + 2 < n; i++)
        cand.insert((uint32_t)s[i] << 16 | (uint32_t)s[i+1] << 8 | s[i+2]);
    if (cand.empty()) return 0.0f;
    int hit = 0;
    for (auto g3 : cand) if (seed_grams.count(g3)) hit++;
    return (float)hit / cand.size();
}
unordered_set<uint32_t> build_seed_grams(const vector<unsigned char>& seed) {
    unordered_set<uint32_t> v;
    for (size_t i = 0; i + 2 < seed.size(); i++)
        v.insert((uint32_t)seed[i] << 16 | (uint32_t)seed[i+1] << 8 | seed[i+2]);
    return v;
}

// ---------------- 评测 ----------------
// 【记】测试集损失：不重叠窗口，每窗 T 个监督位，最多 24 批（384 窗口）控制耗时
double eval_loss(GPT& g, const vector<unsigned char>& test) {
    int nb = min(24, (int)(test.size() / (T + 1) / B));
    if (nb < 1) return -1;
    double loss = 0; long cnt = 0;
    vector<unsigned char> w(B * (T + 1));
    for (int b2 = 0; b2 < nb; b2++) {
        for (int b = 0; b < B; b++) {
            size_t pos = ((size_t)(b2 * B + b) * T) % (test.size() - T - 1);
            for (int t = 0; t <= T; t++) w[b * (T + 1) + t] = test[pos + t];
        }
        loss += forward(g, w.data());
        cnt++;
    }
    return loss / cnt;
}

// ---------------- 自采样 ----------------
// 【记】B 路并行续写：每路一个随机窗口当火种，一次前向每路吐 1 字节
vector<unsigned char> gen_bytes(GPT& g, const vector<unsigned char>& src,
                                int nbytes, float temp, mt19937& rng) {
    vector<unsigned char> out;
    out.reserve(nbytes + B);
    vector<unsigned char> w(B * (T + 1));
    for (int b = 0; b < B; b++) {
        size_t pos = rng() % (src.size() - T - 2);
        for (int t = 0; t < T; t++) w[b * (T + 1) + t] = src[pos + t];
    }
    uniform_real_distribution<double> ud(0, 1);
    while ((int)out.size() < nbytes) {
        forward(g, w.data());
        for (int b = 0; b < B; b++) {
            const float* pr = &g.probs[((size_t)(b * T + T - 1)) * V];
            double q[256], s = 0;   // 【难】温度采样：先幂变换再按 cumulative 抽
            for (int y = 0; y < 256; y++) { q[y] = pow((double)max(pr[y], 1e-12f), 1.0 / temp); s += q[y]; }
            double r = ud(rng) * s;
            int nxt = 0;
            while (r > q[nxt]) { r -= q[nxt]; nxt++; }
            out.push_back(w[b * (T + 1) + T - 1]);   // 收当前末尾字节
            for (int t = 0; t < T - 1; t++) w[b * (T + 1) + t] = w[b * (T + 1) + t + 1];
            w[b * (T + 1) + T - 1] = (unsigned char)nxt;
        }
    }
    out.resize(nbytes);
    return out;
}

// ---------------- 梯度手术工具（D 组） ----------------
// 【记】把 GPT 的全部梯度指针收进一张表，快照/余弦/缩放都靠它，不 hand-write 两遍
vector<vector<float>*> grad_ptrs(GPT& g) {
    vector<vector<float>*> v = {&g.g_E, &g.g_P, &g.lnFg, &g.lnFb, &g.g_WU};
    for (auto& l : g.layers)
        for (auto p : {&l.g_ln1g, &l.g_ln1b, &l.g_Wq, &l.g_bq, &l.g_Wk, &l.g_bk,
                       &l.g_Wv, &l.g_bv, &l.g_Wo, &l.g_bo, &l.g_ln2g, &l.g_ln2b,
                       &l.g_W1, &l.g_b1, &l.g_W2, &l.g_b2})
            v.push_back(p);
    return v;
}
// 组划分：0=嵌入(E,P) 1..L=各层(该层全部梯度) L+1=输出头(WU+顶层LN)
int group_of(int idx) {   // idx 是 grad_ptrs 里的下标
    if (idx <= 1) return 0;                 // g_E, g_P = 嵌入组
    if (idx <= 4) return L + 1;             // g_lnFg, g_lnFb, g_WU = 输出头组
    return (idx - 5) / 16 + 1;              // 之后每层正好 16 个梯度
}
double group_cos(vector<vector<float>*>& A, vector<vector<float>*>& B, vector<vector<float>>& snapA, int grp) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < A.size(); i++) {
        if (group_of((int)i) != grp) continue;
        auto& a = snapA[i]; auto& b = *B[i];
        for (size_t j = 0; j < a.size(); j++) { d += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j]; }
    }
    return d / (sqrt(na) * sqrt(nb) + 1e-12);
}
void scale_group(vector<vector<float>*>& A, vector<vector<float>>& snapA, int grp, float f) {
    for (size_t i = 0; i < A.size(); i++) {
        if (group_of((int)i) != grp) continue;
        auto& dst = *A[i]; auto& src = snapA[i];
        for (size_t j = 0; j < dst.size(); j++) dst[j] = src[j] * f;
    }
}
// ---------------- 候选池 ----------------
const int NCAND = 384;                 // 每次任务切换自采样的候选条数
const int REPLAY_ROWS = 8;             // 批量 16 中 8 行复习 = 1:1 回放比例
const int REFRESH_EVERY = 100;         // 每 100 步重筛一次复习缓冲
void save_pool(const char* path, vector<vector<unsigned char>>& cand, vector<float>& ce) {
    ofstream out(path, ios::binary);
    int32_t n = cand.size(); out.write((char*)&n, 4);
    for (auto& c : cand) out.write((char*)c.data(), T);
    out.write((char*)ce.data(), n * 4);
    out.close();
}
void load_pool(const char* path, vector<vector<unsigned char>>& cand, vector<float>& ce) {
    ifstream in(path, ios::binary);
    int32_t n; in.read((char*)&n, 4);
    cand.resize(n, vector<unsigned char>(T));
    for (auto& c : cand) in.read((char*)c.data(), T);
    ce.resize(n); in.read((char*)ce.data(), n * 4);
    in.close();
}
// 冻结旧副本给候选打分（自困惑度信号，成本一次前向）
vector<float> score_pool(GPT& frozen, vector<vector<unsigned char>>& cand) {
    vector<float> ce(cand.size(), 0.0f);
    vector<unsigned char> w(B * (T + 1));
    for (size_t i = 0; i < cand.size(); i += B) {
        int nb = min(B, (int)cand.size() - (int)i);
        for (int b = 0; b < B; b++) {
            auto& c = cand[min(i + b, cand.size() - 1)];
            for (int t = 0; t < T - 1; t++) w[b * (T + 1) + t] = c[t];
            w[b * (T + 1) + T - 1] = c[T - 1];
        }
        float l = forward(frozen, w.data());
        for (int b = 0; b < nb; b++) ce[i + b] = l;
    }
    return ce;
}

// ---------------- 手术（D 组，微尺度 CPLUS） ----------------
// 【记】s = cos(g_new, g_probe)；s<0 的组整体缩放 g_new *= 1/(1+|s|)（λ=1）
void surgery(GPT& g, GPT& frozen, vector<vector<float>>& snapA, vector<float>& cosLog, FILE* sf, int t) {
    auto gpG = grad_ptrs(g);
    auto gpF = grad_ptrs(frozen);
    // 快照 = 混合批梯度（g_new）
    snapA.resize(gpG.size());
    for (size_t i = 0; i < gpG.size(); i++) snapA[i] = *gpG[i];
    // 【记】probe 梯度由调用方在 frozen 上先算好（顺序：backward(g) -> forward/backward(frozen) -> surgery）
    cosLog.assign(L + 2, 0.0f);
    for (int grp = 0; grp <= L + 1; grp++) {
        double s = group_cos(gpG, gpF, snapA, grp);
        cosLog[grp] = (float)s;
        float f = (s < 0) ? (float)(1.0 / (1.0 + fabs(s))) : 1.0f;
        scale_group(gpG, snapA, grp, f);
    }
    if (sf) {
        fprintf(sf, "t=%d", t);
        for (int grp = 0; grp <= L + 1; grp++) fprintf(sf, " g%d=%.4f", grp, cosLog[grp]);
        fprintf(sf, "\n");
        fflush(sf);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("用法见文件头注释\n"); return 1; }
    string mode = argv[1];
    GPT g;

    // ================= T1：共享分叉点（只训一次） =================
    if (mode == "t1") {
        int steps = atoi(argv[3]);              // 总步数（分段续训时传总目标）
        int step0 = argc > 5 ? atoi(argv[5]) : 0;   // 已完成步数：>0 则读分叉点续训
        if (step0 > 0) loadckpt("t1_fork.bin", g);
        else loadckpt(argv[2], g);
        auto t1 = readbin("t1_train.bin");
        mt19937 rng(42 + step0);
        Adam adam;
        vector<unsigned char> w(B * (T + 1));
        for (int step = step0 + 1; step <= steps; step++) {
            for (int b = 0; b < B; b++) {
                size_t pos = rng() % (t1.size() - T - 1);
                for (int t = 0; t <= T; t++) w[b * (T + 1) + t] = t1[pos + t];
            }
            for (auto& l : g.layers) l.zeroGrad();
            fill(g.g_E.begin(), g.g_E.end(), 0.0f); fill(g.g_P.begin(), g.g_P.end(), 0.0f);
            fill(g.g_lnFg.begin(), g.g_lnFg.end(), 0.0f); fill(g.g_lnFb.begin(), g.g_lnFb.end(), 0.0f);
            fill(g.g_WU.begin(), g.g_WU.end(), 0.0f);
            float loss = forward(g, w.data());
            backward(g, w.data());
            float lr = step < steps * 0.6 ? 0.001f : 0.0003f;
            applyAdam(g, adam, step, lr);
            if (step % 20 == 0) printf("T1 step %d 损失 %.3f\n", step, loss);
            if (step % 100 == 0) saveckpt("t1_fork.bin", g);
        }
        saveckpt("t1_fork.bin", g);
        printf("T1 done -> t1_fork.bin\n");
        return 0;
    }

    // ================= arm：持续学习对照组 =================
    if (mode == "arm") {
        string arm = argv[2];
        int seed = atoi(argv[3]);
        int spd = atoi(argv[4]);            // 每域步数
        bool resume = argc > 5 && string(argv[5]) == "resume";
        const vector<string> ARMS = {"A", "B0", "B1", "B2", "B3", "C", "D"};
        if (find(ARMS.begin(), ARMS.end(), arm) == ARMS.end()) { printf("未知组 %s\n", arm.c_str()); return 1; }
        bool has_replay = arm != "A";
        bool use_pool = arm[0] == 'B' || arm == "D";

        char ckpt[128], statef[128], resf[128];
        sprintf(ckpt, "ckpt_%s_s%d.bin", arm.c_str(), seed);
        sprintf(statef, "state_%s_s%d.txt", arm.c_str(), seed);
        sprintf(resf, "results_%s_s%d.txt", arm.c_str(), seed);

        int k = 2, done = 0;                 // 当前域 / 该域已训步数
        if (resume) {
            FILE* sf = fopen(statef, "r");
            if (sf) { fscanf(sf, "%d %d", &k, &done); fclose(sf); loadckpt(ckpt, g); }
            else { loadckpt("t1_fork.bin", g); }
        } else {
            loadckpt("t1_fork.bin", g);
            auto t1t = readbin("t1_test.bin");
            FILE* rf = fopen(resf, "a");
            fprintf(rf, "EVAL arm=%s seed=%d after=D1 T1=%.4f\n", arm.c_str(), seed, eval_loss(g, t1t));
            fclose(rf);
        }
        mt19937 rng(1000 + seed * 77);
        Adam adam;
        FILE* surgf = (arm == "D") ? fopen((string("surg_") + arm + "_s" + to_string(seed) + ".txt").c_str(), "a") : nullptr;

        for (; k <= 3; k++) {
            char domf[64], testf[64];
            sprintf(domf, "t%d_train.bin", k);
            sprintf(testf, "t%d_test.bin", k);
            auto dom = readbin(domf);
            vector<vector<unsigned char>> pool_cand;
            vector<float> pool_ce, legal, ovl;
            unordered_set<uint32_t> seed_grams;
            vector<unsigned char> real_replay;
            GPT frozen;
            bool frozen_ready = false;

            if (has_replay) {
                // ---- 准备复习源 ----
                if (use_pool) {
                    char poolf[128];
                    if (k == 2) sprintf(poolf, "pool_fork_s%d.bin", seed);
                    else sprintf(poolf, "pool_%s_s%d_d3.bin", arm.c_str(), seed);
                    FILE* test = fopen(poolf, "rb");
                    if (!test) {
                        // 【记】候选池分块续跑：每块 48 条先落 .part 文件，超时被杀也不重来
                        string part = string(poolf) + ".part";
                        int done_n = 0;
                        ifstream pin(part, ios::binary);
                        if (pin) { int32_t dn = 0; pin.read((char*)&dn, 4); done_n = dn; pin.close(); }
                        vector<unsigned char> all(done_n * T);
                        if (done_n > 0) {
                            ifstream pin2(part, ios::binary);
                            int32_t dn; pin2.read((char*)&dn, 4);
                            pin2.read((char*)all.data(), (size_t)done_n * T);
                            pin2.close();
                        }
                        vector<unsigned char> nl;   // 合成提示：换行缓冲，零真实数据泄漏
                        for (int i = 0; i < 2048; i++) nl.push_back(0x0A);
                        const int BLK = 48;
                        while (done_n < NCAND) {
                            printf("候选池 %s: %d/%d\n", poolf, done_n, NCAND); fflush(stdout);
                            auto chunk = gen_bytes(g, nl, BLK * T, 0.8f, rng);
                            all.insert(all.end(), chunk.begin(), chunk.end());
                            done_n += BLK;
                            ofstream pout(part, ios::binary);   // 【易错】整文件重写，别用 append（崩了会混）
                            int32_t dn = min(done_n, NCAND);
                            pout.write((char*)&dn, 4);
                            pout.write((char*)all.data(), (size_t)dn * T);
                            pout.close();
                        }
                        vector<vector<unsigned char>> cv;
                        for (int i = 0; i < NCAND; i++)
                            cv.emplace_back(all.begin() + i * T, all.begin() + (i + 1) * T);
                        GPT fz = g;               // 冻结旧副本 = 切换时刻权重
                        auto ce = score_pool(fz, cv);
                        save_pool(poolf, cv, ce);
                        remove(part.c_str());
                        printf("候选池 %s 完成\n", poolf);
                    } else fclose(test);   // 只探测存在性，别泄漏句柄
                    load_pool(poolf, pool_cand, pool_ce);
                    frozen = g; frozen_ready = true;
                    // 种子银行：上一域开头 4KB（B3 信号专用，见论文 3.2）
                    auto prev = readbin(k == 2 ? "t1_train.bin" : "t2_train.bin");
                    vector<unsigned char> sb(prev.begin(), prev.begin() + 4096);
                    seed_grams = build_seed_grams(sb);
                    legal.resize(pool_cand.size()); ovl.resize(pool_cand.size());
                    for (size_t i = 0; i < pool_cand.size(); i++) {
                        legal[i] = utf8_legality(pool_cand[i].data(), T);
                        ovl[i] = ngram_overlap(pool_cand[i].data(), T, seed_grams);
                    }
                } else {   // C 组：真实回放 = 前面所有域的训练数据
                    real_replay = readbin("t1_train.bin");
                    if (k == 3) {
                        auto t2d = readbin("t2_train.bin");
                        real_replay.insert(real_replay.end(), t2d.begin(), t2d.end());
                    }
                }
            }

            // ---- 筛选（各组差异只允许来自这里） ----
            vector<int> sel;
            if (use_pool) {
                for (size_t i = 0; i < pool_cand.size(); i++) sel.push_back((int)i);
                auto keep = [&](function<bool(int)> ok) {
                    vector<int> ns;
                    for (int i : sel) if (ok(i)) ns.push_back(i);
                    if ((int)ns.size() >= REPLAY_ROWS) sel = ns;   // 筛太狠就放宽，保底 8 条
                };
                if (arm == "B1") keep([&](int i) { return legal[i] >= 0.95f; });
                else if (arm == "B2") {
                    vector<float> s2 = pool_ce; sort(s2.begin(), s2.end());
                    float med = s2[s2.size() / 2];
                    keep([&](int i) { return pool_ce[i] <= med; });
                } else if (arm == "B3") {
                    vector<float> s3 = ovl; sort(s3.begin(), s3.end());
                    float med = s3[s3.size() / 2];
                    keep([&](int i) { return ovl[i] <= med; });
                }   // B0 全收，D 用 B1 的 UTF-8 筛
                if (arm == "D") keep([&](int i) { return legal[i] >= 0.95f; });
            }
            printf("arm=%s seed=%d D%d 开始：复习源就绪（%s）\n", arm.c_str(), seed, k,
                   use_pool ? (to_string(sel.size()) + "/" + to_string(pool_cand.size()) + " 候选入选").c_str()
                            : (to_string(real_replay.size() / 1000000) + "MB 真实回放").c_str());
            fflush(stdout);

            // ---- 训练循环 ----
            vector<unsigned char> w(B * (T + 1));
            vector<vector<float>> snapA;
            int gstep = (k - 2) * spd;
            for (int step = done + 1; step <= spd; step++) {
                gstep = (k - 2) * spd + step;
                for (int b = 0; b < B; b++) {
                    if (has_replay && b < REPLAY_ROWS) {
                        if (use_pool) {
                            auto& c = pool_cand[sel[rng() % sel.size()]];
                            for (int t = 0; t < T - 1; t++) w[b * (T + 1) + t] = c[t];
                            w[b * (T + 1) + T - 1] = c[T - 1];
                        } else {
                            size_t pos = rng() % (real_replay.size() - T - 1);
                            for (int t = 0; t <= T; t++) w[b * (T + 1) + t] = real_replay[pos + t];
                        }
                    } else {
                        size_t pos = rng() % (dom.size() - T - 1);
                        for (int t = 0; t <= T; t++) w[b * (T + 1) + t] = dom[pos + t];
                    }
                }
                for (auto& l : g.layers) l.zeroGrad();
                fill(g.g_E.begin(), g.g_E.end(), 0.0f); fill(g.g_P.begin(), g.g_P.end(), 0.0f);
                fill(g.g_lnFg.begin(), g.g_lnFg.end(), 0.0f); fill(g.g_lnFb.begin(), g.g_lnFb.end(), 0.0f);
                fill(g.g_WU.begin(), g.g_WU.end(), 0.0f);
                forward(g, w.data());
                backward(g, w.data());
                if (arm == "D") {
                    // g_probe：冻结旧副本在复习行上的梯度
                    // 【易错】probe 批 16 行全部用复习行——新域行会污染 probe 梯度
                    vector<unsigned char> wp(B * (T + 1));
                    for (int b = 0; b < B; b++)
                        memcpy(wp.data() + b * (T + 1),
                               w.data() + (b % REPLAY_ROWS) * (T + 1), T + 1);
                    for (auto& l : frozen.layers) l.zeroGrad();
                    fill(frozen.g_E.begin(), frozen.g_E.end(), 0.0f); fill(frozen.g_P.begin(), frozen.g_P.end(), 0.0f);
                    fill(frozen.g_lnFg.begin(), frozen.g_lnFg.end(), 0.0f); fill(frozen.g_lnFb.begin(), frozen.g_lnFb.end(), 0.0f);
                    fill(frozen.g_WU.begin(), frozen.g_WU.end(), 0.0f);
                    forward(frozen, wp.data());
                    backward(frozen, wp.data());
                    vector<float> cosLog;
                    surgery(g, frozen, snapA, cosLog, (gstep % 20 == 0) ? surgf : nullptr, gstep);
                }
                applyAdam(g, adam, gstep, 5e-4f);   // 【记】持续学习用较小学习率，防冲掉旧知识
                if (step % 20 == 0) printf("D%d step %d/%d\n", k, step, spd);
                if (step % 100 == 0) {
                    saveckpt(ckpt, g);
                    FILE* sf = fopen(statef, "w"); fprintf(sf, "%d %d\n", k, step); fclose(sf);
                }
            }
            // ---- 域尾评测：所有已见域的测试集 ----
            {
                FILE* rf = fopen(resf, "a");
                string line = "EVAL arm=" + arm + " seed=" + to_string(seed) + " after=D" + to_string(k);
                for (int d = 1; d <= k; d++) {
                    char tf[64]; sprintf(tf, "t%d_test.bin", d);
                    auto tv = readbin(tf);
                    char buf[64]; sprintf(buf, " T%d=%.4f", d, eval_loss(g, tv));
                    line += buf;
                }
                fprintf(rf, "%s\n", line.c_str());
                fclose(rf);
            }
            printf("D%d done. %s\n", k, resf);
            done = 0;
            FILE* sf = fopen(statef, "w"); fprintf(sf, "%d %d\n", k + 1, 0); fclose(sf);
        }
        saveckpt(ckpt, g);
        printf("arm %s seed %d 全部完成\n", arm.c_str(), seed);
        return 0;
    }

    // ================= 坍缩实验三件套 =================
    if (mode == "gen") {   // gen <存档> <提示源> <输出> <字节数> <种子> <温度>
        loadckpt(argv[2], g);
        auto src = readbin(argv[3]);
        mt19937 rng(atoi(argv[6]));
        auto out = gen_bytes(g, src, atoi(argv[4]), atof(argv[7]), rng);
        FILE* f = fopen(argv[5], "wb"); fwrite(out.data(), 1, out.size(), f); fclose(f);
        printf("已生成 %s (%zu 字节)\n", argv[5], out.size());
        return 0;
    }
    if (mode == "trainfile") {   // trainfile <存档> <语料> <步数> <新存档> <种子>
        loadckpt(argv[2], g);
        auto data = readbin(argv[3]);
        int steps = atoi(argv[4]);
        mt19937 rng(atoi(argv[6]));
        Adam adam;
        vector<unsigned char> w(B * (T + 1));
        for (int step = 1; step <= steps; step++) {
            for (int b = 0; b < B; b++) {
                size_t pos = rng() % (data.size() - T - 1);
                for (int t = 0; t <= T; t++) w[b * (T + 1) + t] = data[pos + t];
            }
            for (auto& l : g.layers) l.zeroGrad();
            fill(g.g_E.begin(), g.g_E.end(), 0.0f); fill(g.g_P.begin(), g.g_P.end(), 0.0f);
            fill(g.g_lnFg.begin(), g.g_lnFg.end(), 0.0f); fill(g.g_lnFb.begin(), g.g_lnFb.end(), 0.0f);
            fill(g.g_WU.begin(), g.g_WU.end(), 0.0f);
            forward(g, w.data());
            backward(g, w.data());
            applyAdam(g, adam, step, 5e-4f);
            if (step % 200 == 0) { printf("step %d\n", step); saveckpt(argv[5], g); }
        }
        saveckpt(argv[5], g);
        printf("trainfile done -> %s\n", argv[5]);
        return 0;
    }
    if (mode == "evalfile") {   // evalfile <存档> <语料>
        loadckpt(argv[2], g);
        auto data = readbin(argv[3]);
        printf("LOSS %.4f\n", eval_loss(g, data));
        return 0;
    }
    printf("未知模式 %s\n", mode.c_str());
    return 1;
}
