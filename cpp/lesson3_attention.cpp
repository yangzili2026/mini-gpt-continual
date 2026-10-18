// ============================================================
// C++ 造大语言模型 · 第3课：手写掩码注意力 + 反向传播训练
// ------------------------------------------------------------
// 第2课的 bigram 只有一格记忆。这一课给模型装上"短期记忆"：
//
//   注意力机制三句话：
//   1. 每个字节查表变成一个 D 维向量
//   2. 用 Q·K 点积算"彼此有多相关"，softmax 成注意力权重
//   3. 按权重对 V 加权求和 —— 这就是"看一眼上下文"
//
// 本课两个硬核内容：
//   A. 注意力前向传播（forward）—— 看图说话就能写出来
//   B. 反向传播（backward）—— 用链式法则推出每个参数的梯度公式，
//      写完立刻用"数值梯度"抽查验证（程序自带梯度检查，不可能悄悄错）
//
// 模型：窗口 T=8，维度 D=16，单头注意力。
// ============================================================
// 标注说明：【难】=重点看注释 【易错】=容易写错 【记】=照抄即可
// ============================================================
#include <bits/stdc++.h>   // 万能头
using namespace std;

const int T = 8;              // 记忆长度：看前面 8 个字节
const int D = 24;             // 每个字节的向量维度
const double SQRT_D = sqrt((double)D);

// ---------- 模型参数 和 对应的梯度容器 ----------
struct Mats {
    vector<vector<double>> E;    // 嵌入表 256×D：字节 -> 向量
    vector<vector<double>> Wq;   // 查询矩阵 D×D
    vector<vector<double>> Wk;   // 键矩阵   D×D
    vector<vector<double>> Wv;   // 值矩阵   D×D
    vector<vector<double>> U;    // 输出矩阵 D×256：向量 -> 256 个分数
};

Mats newMats(int mode, mt19937& rng) {
    Mats m;
    auto make = [&](int r, int c) {
        vector<vector<double>> v(r, vector<double>(c));
        if (mode == 0) {   // 参数：小随机数
            uniform_real_distribution<double> u(-0.02, 0.02);
            for (auto& row : v) for (auto& x : row) x = u(rng);
        }                  // mode 1：全 0（梯度容器）
        return v;
    };
    m.E = make(256, D); m.Wq = make(D, D); m.Wk = make(D, D);
    m.Wv = make(D, D); m.U = make(D, 256);
    return m;
}

// ============================================================
// 前向 + 反向（本课核心）
// 我们只关心"预测下一个字节"，所以注意力只用最后一个位置的查询，
// 因果掩码对这个查询是自动满足的（它右边没有字可看）。
// ============================================================
double forward_backward(const Mats& m, const unsigned char* ctx, int target,
                        Mats* g, vector<double>* attnOut = nullptr,
                        double* probsOut = nullptr, bool needGrad = true) {
    // ---------- 前向 ----------
    double x[T][D];
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) x[i][d] = m.E[ctx[i]][d];

    double Q[D] = {0};
    for (int d = 0; d < D; d++)
        for (int e = 0; e < D; e++) Q[d] += x[T-1][e] * m.Wq[e][d];

    double K[T][D] = {}, V[T][D] = {};
    for (int j = 0; j < T; j++)
        for (int d = 0; d < D; d++)
            for (int e = 0; e < D; e++) {
                K[j][d] += x[j][e] * m.Wk[e][d];
                V[j][d] += x[j][e] * m.Wv[e][d];
            }

    // 注意力分数 + softmax（最后一个位置可以看全部 T 个位置）
    double s[T], A[T], mx = -1e18, sum = 0;
    for (int j = 0; j < T; j++) {
        s[j] = 0;
        for (int d = 0; d < D; d++) s[j] += Q[d] * K[j][d];
        s[j] /= SQRT_D;
        mx = max(mx, s[j]);
    }
    for (int j = 0; j < T; j++) { A[j] = exp(s[j] - mx); sum += A[j]; }
    for (int j = 0; j < T; j++) A[j] /= sum;
    if (attnOut) *attnOut = vector<double>(A, A + T);

    double o[D] = {0};
    for (int j = 0; j < T; j++)
        for (int d = 0; d < D; d++) o[d] += A[j] * V[j][d];

    double logits[256], mx2 = -1e18, sum2 = 0, p[256];
    for (int y = 0; y < 256; y++) {
        logits[y] = 0;
        for (int d = 0; d < D; d++) logits[y] += o[d] * m.U[d][y];
        mx2 = max(mx2, logits[y]);
    }
    for (int y = 0; y < 256; y++) { p[y] = exp(logits[y] - mx2); sum2 += p[y]; }
    for (int y = 0; y < 256; y++) p[y] /= sum2;

    double loss = -log(max(p[target], 1e-12));
    if (probsOut) for (int y = 0; y < 256; y++) probsOut[y] = p[y];
    if (!needGrad) return loss;

    // ---------- 反向（链式法则，每一步都很短） ----------
    // 1) softmax+交叉熵的梯度：概率 - 正确答案
    double glogits[256];
    for (int y = 0; y < 256; y++) glogits[y] = p[y] - (y == target ? 1.0 : 0.0);

    // 2) 输出层：dU = o ⊗ glogits；并反传得到 go
    double go[D] = {0};
    for (int d = 0; d < D; d++) {
        for (int y = 0; y < 256; y++) {
            g->U[d][y] += o[d] * glogits[y];          // U 的梯度
            go[d] += m.U[d][y] * glogits[y];         // 传到 o
        }
    }

    // 3) 注意力权重 A 的梯度：gA[j] = go · V[j]
    double gA[T], dotAV = 0;
    for (int j = 0; j < T; j++) {
        gA[j] = 0;
        for (int d = 0; d < D; d++) gA[j] += go[d] * V[j][d];
        dotAV += A[j] * gA[j];
    }
    // 【难】softmax 反传：每个分数的梯度要减掉"按注意力的加权平均"
    double gs[T];
    for (int j = 0; j < T; j++) gs[j] = A[j] * (gA[j] - dotAV);

    // 4) 分数 -> Q、K 的梯度
    double gQ[D] = {}, gK[T][D] = {}, gV[T][D] = {};
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < D; d++) {
            gQ[d] += gs[j] * K[j][d] / SQRT_D;       // Q 的梯度
            gK[j][d] = gs[j] * Q[d] / SQRT_D;        // K 的梯度
            gV[j][d] = A[j] * go[d];                 // V 的梯度
        }
    }

    // 5) Q/K/V -> 三个矩阵的梯度（外积：x 的转置 × 上游梯度）
    for (int e = 0; e < D; e++)
        for (int d = 0; d < D; d++) {
            g->Wq[e][d] += x[T-1][e] * gQ[d];
            for (int j = 0; j < T; j++) {
                g->Wk[e][d] += x[j][e] * gK[j][d];
                g->Wv[e][d] += x[j][e] * gV[j][d];
            }
        }

    // 6) 传到 x，再传进嵌入表 E
    for (int j = 0; j < T; j++) {
        double gx[D] = {0};
        for (int d = 0; d < D; d++)
            for (int e = 0; e < D; e++)
                gx[d] += gK[j][e] * m.Wk[d][e] + gV[j][e] * m.Wv[d][e];
        for (int d = 0; d < D; d++) g->E[ctx[j]][d] += gx[d];
    }
    for (int d = 0; d < D; d++) {
        double acc = 0;
        for (int e = 0; e < D; e++) acc += gQ[e] * m.Wq[d][e];
        g->E[ctx[T-1]][d] += acc;     // 最后一个位置还背着 Q 的梯度
    }
    return loss;
}

// ---------- 数值梯度检查：抽 30 个参数和解析梯度对答案 ----------
void gradcheck(Mats& m, mt19937& rng, const vector<unsigned char>& data) {
    uniform_real_distribution<double> err;
    double worst = 0;
    for (int t = 0; t < 30; t++) {
        int pos = T + rng() % (data.size() - T - 1);
        unsigned char ctx[T];
        for (int i = 0; i < T; i++) ctx[i] = data[pos - T + i];
        int target = data[pos];

        Mats g = newMats(1, rng);
        forward_backward(m, ctx, target, &g);

        // 随机抽一个参数位置
        int which = rng() % 5, r, c;
        vector<vector<double>>* M;
        vector<vector<double>>* G;
        if (which == 0) { r = rng() % 256; c = rng() % D;      M = &m.E;  G = &g.E;  }
        else if (which == 1) { r = rng() % D; c = rng() % D;   M = &m.Wq; G = &g.Wq; }
        else if (which == 2) { r = rng() % D; c = rng() % D;   M = &m.Wk; G = &g.Wk; }
        else if (which == 3) { r = rng() % D; c = rng() % D;   M = &m.Wv; G = &g.Wv; }
        else { r = rng() % D; c = rng() % 256;                 M = &m.U;  G = &g.U;  }

        const double EPS = 1e-5;
        (*M)[r][c] += EPS;
        Mats dummy = newMats(1, rng);
        double l1 = forward_backward(m, ctx, target, nullptr, nullptr, nullptr, false);
        (*M)[r][c] -= 2 * EPS;
        double l2 = forward_backward(m, ctx, target, nullptr, nullptr, nullptr, false);
        (*M)[r][c] += EPS;
        double num = (l1 - l2) / (2 * EPS);          // 数值梯度（中心差分）
        double ana = (*G)[r][c];                      // 解析梯度（我们推的）
        worst = max(worst, fabs(num - ana));
    }
    cout << "梯度检查：30 个抽查点，解析 vs 数值 最大偏差 = " << scientific
         << worst << "  （小于 1e-6 即合格）" << endl;
}

// ---------- Adam 优化器：对学习率不敏感，皮实 ----------
struct Adam {
    Mats mm, vv;   // 一阶/二阶动量
    Adam(mt19937& rng) : mm(newMats(1, rng)), vv(newMats(1, rng)) {}
};
void adamStep(Mats& m, Mats& g, Adam& a, int t, double lr) {
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    auto step = [&](vector<vector<double>>& M, vector<vector<double>>& G,
                    vector<vector<double>>& M1, vector<vector<double>>& V1) {
        double c1 = 1 - pow(b1, t), c2 = 1 - pow(b2, t);
        for (size_t r = 0; r < M.size(); r++)
            for (size_t c = 0; c < M[0].size(); c++) {
                M1[r][c] = b1 * M1[r][c] + (1 - b1) * G[r][c];
                V1[r][c] = b2 * V1[r][c] + (1 - b2) * G[r][c] * G[r][c];
                M[r][c] -= lr * (M1[r][c] / c1) / (sqrt(V1[r][c] / c2) + eps);
            }
    };
    step(m.E, g.E, a.mm.E, a.vv.E);
    step(m.Wq, g.Wq, a.mm.Wq, a.vv.Wq);
    step(m.Wk, g.Wk, a.mm.Wk, a.vv.Wk);
    step(m.Wv, g.Wv, a.mm.Wv, a.vv.Wv);
    step(m.U, g.U, a.mm.U, a.vv.U);
}

int main() {
    // ---------- 1. 读四大名著 ----------
    const char* files[] = {"hongloumeng.txt", "sanguoyanyi.txt",
                           "shuihuzhuan.txt", "xiyouji.txt"};
    vector<unsigned char> data;
    for (auto fn : files) {
        ifstream fin(fn, ios::binary);   // 【易错】ios::binary
        vector<unsigned char> part((istreambuf_iterator<char>(fin)),
                                   istreambuf_iterator<char>());
        data.insert(data.end(), part.begin(), part.end());
        fin.close();
    }
    cout << "训练语料: 四大名著，共 " << data.size() << " 个字节" << endl;

    mt19937 rng(42);
    Mats m = newMats(0, rng);
    Adam adam(rng);

    gradcheck(m, rng, data);   // 先证明反向传播没推错，再开始训练

    // ---------- 2. 训练 ----------
    const int STEPS = 15000, BATCH = 32;
    for (int step = 1; step <= STEPS; step++) {
        Mats g = newMats(1, rng);
        double loss = 0;
        for (int b = 0; b < BATCH; b++) {
            int pos = T + rng() % (data.size() - T - 1);
            unsigned char ctx[T];
            for (int i = 0; i < T; i++) ctx[i] = data[pos - T + i];
            loss += forward_backward(m, ctx, data[pos], &g);
        }
        // 梯度取平均（Adam 不怕梯度大小，但求平均是标准做法）
        for (auto& row : g.E) for (auto& v : row) v /= BATCH;
        for (auto* mat : {&g.Wq, &g.Wk, &g.Wv, &g.U})
            for (auto& row : *mat) for (auto& v : row) v /= BATCH;
        double lr = step < 8000 ? 0.02 : 0.006;   // 先大步探索，后小步精修
        adamStep(m, g, adam, step, lr);

        if (step % 1000 == 0)
            cout << "step " << step << "  平均损失 = " << fixed << setprecision(3)
                 << loss / BATCH << "  （随机瞎猜 5.545，bigram 约 2.8）" << endl;
    }

    // ---------- 3. 注意力热力图：模型在"笑道"时看谁 ----------
    int found = -1;
    for (size_t i = 8; i + 6 < data.size(); i++)
        if (data[i]==231 && data[i+1]==172 && data[i+2]==145 &&
            data[i+3]==233 && data[i+4]==129 && data[i+5]==147) { found = i; break; }

    if (found >= 0) {
        unsigned char w[T];
        for (int i = 0; i < T; i++) w[i] = data[found - 2 + i];
        vector<double> attn;
        Mats dummy = newMats(1, rng);
        forward_backward(m, w, data[found + 4], nullptr, &attn, nullptr, false);
        cout << endl << "预测\"道\"字后面的字节时，模型的目光落在窗口 8 个位置上的比例：" << endl;
        for (int j = 0; j < T; j++) {
            char tag = (j == T - 1) ? '<' : ' ';   // 标记最后一个位置（"道"字本身）
            printf("  位置%d 字节%3d %c %.3f %s\n", j, w[j], tag, attn[j],
                   string((int)(attn[j] * 40), '#').c_str());
        }
    }

    // ---------- 4. 期末考 ----------
    double loss = 0; long long hit = 0, n = 0;
    for (size_t pos = T; pos < 150000 && pos < data.size(); pos++) {
        unsigned char w[T];
        for (int i = 0; i < T; i++) w[i] = data[pos - T + i];
        double probs[256];
        forward_backward(m, w, data[pos], nullptr, nullptr, probs, false);
        loss += -log(max(probs[data[pos]], 1e-12));
        int best = 0;
        for (int y = 1; y < 256; y++) if (probs[y] > probs[best]) best = y;
        if (best == (int)data[pos]) hit++;
        n++;
    }
    cout << endl << "期末考[红楼梦前15万窗口] 平均损失 = " << loss / n
         << "  预测准确率 " << fixed << setprecision(1) << 100.0 * hit / n
         << "%  （bigram: 损失 2.731 / 准确率 25.9%）" << endl;

    // ---------- 5. 仿写 300 字节（滑窗 + 按模型概率采样） ----------
    unsigned char cur[T];
    for (int i = 0; i < T; i++) cur[i] = data[found - 2 + i];
    string out;
    for (int gStep = 0; gStep < 300; gStep++) {
        double q[256], s = 0;
        forward_backward(m, cur, 0, nullptr, nullptr, q, false);  // 只要概率分布
        // 【难】温度采样：给概率开 1/温度 次方，温度<1 时大概率更大、小概率更小，
        // 过滤掉"似是而非"的生僻组合（真实大模型也这么干）
        for (int y = 0; y < 256; y++) q[y] = pow(q[y], 1.0 / 0.7);
        for (int y = 0; y < 256; y++) s += q[y];   // 【易错】归一化常数，漏了会一直选中字节0
        double r = uniform_real_distribution<>(0, 1)(rng) * s;
        int nxt = 0;
        while (r > q[nxt]) { r -= q[nxt]; nxt++; }
        out.push_back((char)cur[0]);
        for (int i = 0; i < T - 1; i++) cur[i] = cur[i + 1];
        cur[T - 1] = (unsigned char)nxt;
    }
    out.append((char*)cur, T);
    ofstream fout("gen3.bin", ios::binary);
    fout.write(out.data(), out.size());
    fout.close();
    cout << "仿写已存 gen3.bin（开头 8 字节是书中原文，后面是模型续写）" << endl;
    return 0;
}
