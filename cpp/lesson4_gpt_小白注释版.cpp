// ============================================================================
// C++ 造大语言模型 · 第4课：完整迷你 GPT · 小白逐行注释版
// ----------------------------------------------------------------------------
// 这份文件和 lesson4_gpt.cpp 的【代码一模一样】，只是每一部分都加了
// 详细注释。带标记的注释是你要求的三级标注：
//   【难】  有点复杂，第一次见很正常，慢慢品
//   【易错】 容易写错/我们曾经踩过坑的地方
//   【记】  需要记住的结论
//
// 阅读地图（建议按这个顺序读）：
//   第1区：超参数        —— 模型的"身体指标"，先背下来
//   第2区：矩阵工具      —— 只有乘法，但是一切的地基
//   第3区：LayerNorm     —— "值日生"：把每行数字整理成标准尺度
//   第4区：模型结构      —— 每个零件叫什么、装在哪
//   第5区：前向传播      —— 一个字如何流过整个网络变成预测
//   第6区：反向传播      —— 考砸了怎么把"错在哪"一层层传回去
//   第7区：Adam          —— 带惯性的改错本
//   第8区：主流程        —— 训练循环 + 期末考 + 仿写
// ============================================================================
#include <bits/stdc++.h>
using namespace std;

// ============================================================================
// 第1区：超参数 —— 模型的"身体指标"
// ============================================================================
// 【记】这 8 个数字决定了整个模型的大小。改这里就是在改模型本身。
const int V = 256;    // 词表大小 = 256 个字节（0~255 全都有，一个不漏）
const int T = 32;     // 上下文窗口：每次让模型看 32 个字节，然后猜第 33 个
const int D = 128;    // 向量维度：每个字节在模型内部是一个 128 维向量
const int H = 4;      // 注意力头数：4 个"小组"分别从 4 个角度互相看
const int NH = D / H; // 每个头分到的维度 = 128/4 = 32
                      // 【记】头不是 4 份独立的小模型，而是把 128 维切成 4 段，
                      //        每段 32 维各自做注意力，最后拼回 128 维
const int L = 4;      // 层数：同样的"注意力+前馈"模块叠 4 层（论文里是 6 层）
const int F = 512;    // 前馈网络中间层宽度 = 4×128，和论文的比例一致
const int B = 16;     // 批量：每次训练同时处理 16 个窗口（样本）
const int BT = B * T; // = 512，批量里字节位置的总数（16×32），到处都要用
const float SQRT_NH = sqrtf((float)NH);   // √32，注意力公式里除以它，防止数字太大

// ============================================================================
// 第2区：矩阵工具 —— 整份代码只有这 5 个小函数在干"体力活"
// ============================================================================
// 【难】矩阵乘法是什么？C = A @ B，就是 C 的第 i 行第 j 列 =
//      A 的第 i 行 和 B 的第 j 列 对应相乘再全部加起来。
//      大学里叫线性代数，你现在只需要会算：1×3 + 2×4 + 3×5 = 26 这种。
//
// 为什么还要 gemm_bt、gemm_at 两个变体？因为矩阵乘法的三种"转置姿势"
// 在反向传播里都会出现。它们只是换了一种取数方式，算的还是同一件事。
//
// 【记】所有矩阵都是"行优先"平铺的一维数组：第 i 行第 j 列 = X[i*列数 + j]

// 姿势一：C[m×n] = A[m×k] @ B[k×n]（最普通的乘法）
void gemm(const float* A, const float* Bm, float* C, int m, int k, int n) {
    memset(C, 0, sizeof(float) * (size_t)m * n);          // 先清零，准备累加
    for (int i = 0; i < m; i++)                           // 逐行
        for (int l = 0; l < k; l++) {                     // 沿着"被乘的那一行"
            float a = A[i * k + l];                       // 先取出 A[i][l]，省得反复查
            const float* Brow = Bm + (size_t)l * n;       // B 的第 l 行
            float* Crow = C + (size_t)i * n;              // C 的第 i 行
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];   // 整行累加
        }
}

// 姿势二：C[m×n] = A[m×k] @ B[n×k]^T（B 躺着进来，第 j 列当行用）
// 用途：反向传播里乘 W^T，把梯度从输出端送回输入端
void gemm_bt(const float* A, const float* Bm, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[j * k + l];
            C[i * n + j] = s;
        }
}

// 姿势三：C[m×n] = A[k×m]^T @ B[k×n]（A 躺着进来）
// 用途：反向传播里算"权重矩阵自己的梯度"
void gemm_at(const float* A, const float* Bm, float* C, int m, int k, int n) {
    memset(C, 0, sizeof(float) * (size_t)m * n);
    for (int l = 0; l < k; l++)
        for (int i = 0; i < m; i++) {
            float a = A[(size_t)l * m + i];
            const float* Brow = Bm + (size_t)l * n;
            float* Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
}

// 给矩阵每一行加上 bias（偏置向量）
// 比喻：bias 是每个神经元自带的"门槛高低"，与输入无关
void bias_add(float* X, const float* bias, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) X[i * cols + j] += bias[j];
}

// 反向传播专用：把矩阵每一列加起来，得到 bias 的梯度
// 因为每个样本都给 bias 贡献了一份梯度，所以要"求和"而不是平均
void colsum(const float* X, float* bias, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        float s = 0;
        for (int i = 0; i < rows; i++) s += X[i * cols + j];
        bias[j] += s;
    }
}

// ============================================================================
// 第3区：LayerNorm —— "值日生"
// ============================================================================
// 【难】LayerNorm（层归一化）：把一行里的 D=128 个数，
//      先减去平均值（让这行以 0 为中心），
//      再除以标准差（让这行的波动幅度变成 1），
//      最后乘 g 加 b（给模型留两个可以学习的旋钮，决定"到底要不要真归一化"）。
//
// 为什么需要它？想象 4 层网络层层放大数字，滚到第 4 层数字大得离谱
// （第3课就因此 NaN 爆炸过）。LayerNorm 就像值日生：不管你们闹多凶，
// 每进一层之前先把每行整理回"标准身材"，训练就稳了。
//
// 输入：x 原始数据，g/b 可学习的旋钮（gamma/beta）
// 输出：y 归一化后的结果；同时把 xhat（标准化结果）和 rstd（1/标准差）
//      存起来，因为反向传播还要用 —— 【记】前向存缓存，反向复用，这是铁律
void layernorm_fwd(const float* x, const float* g, const float* b,
                   float* xhat, float* rstd, float* y, int rows) {
    for (int i = 0; i < rows; i++) {              // 一行一行处理（每行 D 个数）
        const float* xr = x + (size_t)i * D;
        float mean = 0, var = 0;
        for (int j = 0; j < D; j++) mean += xr[j];   // 第一步：求和
        mean /= D;                                   // 平均值
        for (int j = 0; j < D; j++) var += (xr[j] - mean) * (xr[j] - mean);  // 方差
        float rs = 1.0f / sqrtf(var / D + 1e-5f);    // 1/标准差，加 1e-5 防除以 0
        rstd[i] = rs;                                // 存起来，反向要用
        for (int j = 0; j < D; j++) {
            xhat[i * D + j] = (xr[j] - mean) * rs;   // 标准化：均值0、方差1
            y[i * D + j] = g[j] * xhat[i * D + j] + b[j];   // 乘旋钮、加旋钮
        }
    }
}

// 【难】LayerNorm 的反向传播：已知"输出的梯度 dy"，求"输入的梯度 dx"
//      和两个旋钮的梯度 dg、db。
//      公式的直觉：一行里的 128 个数互相牵连（减的是同一个均值，
//      除的是同一个标准差），所以每个数的梯度都收到全行的影响。
//      具体推导是大学的微积分，现在记住形状即可：
//      dx = rstd × (g·dy − 全行均值(g·dy) − xhat × 全行均值(g·dy·xhat))
void layernorm_bwd(const float* xhat, const float* rstd, const float* g,
                   const float* dy, float* dx, float* dg, float* db, int rows) {
    for (int i = 0; i < rows; i++) {
        const float* hr = xhat + (size_t)i * D;
        const float* dyr = dy + (size_t)i * D;
        float rs = rstd[i];
        float dsum = 0, ddot = 0;
        for (int j = 0; j < D; j++) {
            dg[j] += dyr[j] * hr[j];    // 旋钮 g 的梯度：输出梯度 × 标准化值
            db[j] += dyr[j];            // 旋钮 b 的梯度：输出梯度直接累加
            dsum += dyr[j] * g[j];      // 为 dx 公式收集材料
            ddot += dyr[j] * g[j] * hr[j];
        }
        dsum /= D; ddot /= D;           // 全行均值
        for (int j = 0; j < D; j++)
            dx[i * D + j] = rs * (dyr[j] * g[j] - dsum - hr[j] * ddot);
    }
}

// ============================================================================
// 第4区：模型结构 —— 零件清单
// ============================================================================
// 一层 Transformer = 两小段，每段都是"先整理、再加工、最后把原样加回去"：
//
//   段1（注意力）：LN -> Q/K/V -> 多头互相看 -> 输出投影 -> +原样（残差）
//   段2（前馈）：  LN -> 128升512 -> ReLU -> 512降128 -> +原样（残差）
//
// "残差"（+原样）是什么？x + f(x)。意思是：加工结果只是"修改意见"，
// 原信息始终保留。这样叠很多层也不会把原始信号弄丢 —— 论文里的 Add。
struct Layer {
    // ---- 参数（模型要学习的"知识"，训练就是改这些数）----
    vector<float> ln1g = vector<float>(D), ln1b = vector<float>(D),
                  Wq = vector<float>(D * D), bq = vector<float>(D),
                  Wk = vector<float>(D * D), bk = vector<float>(D),
                  Wv = vector<float>(D * D), bv = vector<float>(D),
                  Wo = vector<float>(D * D), bo = vector<float>(D),
                  ln2g = vector<float>(D), ln2b = vector<float>(D),
                  W1 = vector<float>(D * F), b1 = vector<float>(F),
                  W2 = vector<float>(F * D), b2 = vector<float>(D);
    // 逐个解释：
    //   ln1g/ln1b 段1的值日生旋钮；ln2g/ln2b 段2的值日生旋钮（各 128 个）
    //   Wq/bq  查询矩阵：每个字问"我要找什么"（128×128 + 128）
    //   Wk/bk  键矩阵：每个字挂出"我是什么"的门牌号
    //   Wv/bv  值矩阵：每个字口袋里真正想传递的内容
    //   Wo/bo  注意力结果出来后，再加工一次的"翻译官"
    //   W1/b1  前馈第一层：128 维升到 512 维（把信息摊开）
    //   W2/b2  前馈第二层：512 维压回 128 维（提炼出结论）

    // ---- 梯度（每个参数的"修改意见"，形状和参数一一对应）----
    vector<float> g_ln1g = vector<float>(D), g_ln1b = vector<float>(D),
                  g_Wq = vector<float>(D * D), g_bq = vector<float>(D),
                  g_Wk = vector<float>(D * D), g_bk = vector<float>(D),
                  g_Wv = vector<float>(D * D), g_bv = vector<float>(D),
                  g_Wo = vector<float>(D * D), g_bo = vector<float>(D),
                  g_ln2g = vector<float>(D), g_ln2b = vector<float>(D),
                  g_W1 = vector<float>(D * F), g_b1 = vector<float>(F),
                  g_W2 = vector<float>(F * D), g_b2 = vector<float>(D);

    // ---- 前向缓存（反向传播要重用的"草稿纸"）----
    vector<float> x_in, xhat1, rstd1, y1, q, k, v, att, ctx,
                  x_in2, xhat2, rstd2, y2, hid, ffn;
    Layer() {
        x_in.resize(BT * D); xhat1.resize(BT * D); rstd1.resize(BT);
        y1.resize(BT * D);
        q.resize(BT * D); k.resize(BT * D); v.resize(BT * D);
        att.resize((size_t)B * H * T * T); ctx.resize(BT * D);   // att: 注意力分数表
        x_in2.resize(BT * D); xhat2.resize(BT * D); rstd2.resize(BT);
        y2.resize(BT * D);
        hid.resize((size_t)BT * F); ffn.resize(BT * D);
    }
    void zeroGrad() {   // 每个训练步开头，把上一轮的意见全部清零
        for (auto p : {&g_ln1g, &g_ln1b, &g_Wq, &g_bq, &g_Wk, &g_bk, &g_Wv, &g_bv,
                       &g_Wo, &g_bo, &g_ln2g, &g_ln2b, &g_W1, &g_b1, &g_W2, &g_b2})
            fill(p->begin(), p->end(), 0.0f);
    }
};

// 整个模型 = 嵌入表 + 位置表 + 顶层归一化 + 输出头 + L 个 Layer
struct GPT {
    vector<float> E = vector<float>(V * D), P = vector<float>(T * D),
                  lnFg = vector<float>(D), lnFb = vector<float>(D),
                  WU = vector<float>(D * V);
    //   E    嵌入表：256 个字 × 每个 128 维 = 查字典（第2课的大 lookup table）
    //   P    位置表：32 个座位 × 每个 128 维 = 告诉模型"你在第几个字"
    //        【记】模型只看向量不知道顺序，顺序全靠这张表喂进去
    //   lnFg/lnFb  顶层值日生旋钮（输出头之前最后一次整理）
    //   WU   输出头：128 维 -> 256 类（"下一个字是 0~255 中哪个"的打分）

    vector<float> g_E = vector<float>(V * D), g_P = vector<float>(T * D),
                  g_lnFg = vector<float>(D), g_lnFb = vector<float>(D),
                  g_WU = vector<float>(D * V);
    // 上面这 5 个的梯度

    vector<float> x, xhatf, rstdf, yf, probs;
    //   x      数据主通道：当前每个字的 128 维向量（512 行 × 128）
    //   probs  最终概率表：512 行 × 256 列，每行是"下一个字"的概率分布

    vector<Layer> layers{L};    // 【记】{L} 是 C++11 写法：直接造好 4 层
    GPT() {
        x.resize(BT * D); xhatf.resize(BT * D); rstdf.resize(BT);
        yf.resize(BT * D); probs.resize((size_t)BT * V);
    }
    long long nparams() {   // 数一数一共有多少个要学的数（约 86 万）
        long long n = E.size() + P.size() + lnFg.size() + lnFb.size() + WU.size();
        for (auto& l : layers) n += l.ln1g.size() * 2 + l.Wq.size() * 4 + l.bq.size() * 4
            + l.ln2g.size() * 2 + l.W1.size() + l.b1.size() + l.W2.size() + l.b2.size();
        return n;
    }
};

// ---------------- 初始化：给所有参数一个起点 ----------------
// 【记】起点必须是"小小的随机数"。全 0 会让所有神经元永远同步更新，
//        永远学不会不同的东西（对称性破缺，大学会细讲）。
void initGPT(GPT& g, mt19937& rng) {
    normal_distribution<float> nd(0.0f, 0.02f);   // 均值0、标准差0.02 的正态分布
    auto filln = [&](vector<float>& v) {
        for (auto& x : v) x = nd(rng);
    };
    filln(g.E); filln(g.P); filln(g.WU);
    for (auto& l : g.layers) {
        filln(l.Wq); filln(l.Wk); filln(l.Wv); filln(l.Wo);
        filln(l.W1); filln(l.W2);
        fill(l.ln1g.begin(), l.ln1g.end(), 1.0f);   // 【记】LN 旋钮 g 从 1 开始
        fill(l.ln2g.begin(), l.ln2g.end(), 1.0f);   // （等于"先原样通过"，让模型自己决定调多少）
        fill(l.ln1b.begin(), l.ln1b.end(), 0.0f); fill(l.ln2b.begin(), l.ln2b.end(), 0.0f);
        fill(l.bq.begin(), l.bq.end(), 0.0f); fill(l.bk.begin(), l.bk.end(), 0.0f);
        fill(l.bv.begin(), l.bv.end(), 0.0f); fill(l.bo.begin(), l.bo.end(), 0.0f);
        fill(l.b1.begin(), l.b1.end(), 0.0f); fill(l.b2.begin(), l.b2.end(), 0.0f);
    }
    fill(g.lnFg.begin(), g.lnFg.end(), 1.0f);
    fill(g.lnFb.begin(), g.lnFb.end(), 0.0f);
}

// ============================================================================
// 第5区：前向传播 —— 一个字如何流过整个网络
// ----------------------------------------------------------------------------
// toks：这一批 16×(32+1) 个字节。
// 【易错】为什么是 33 个而不是 32 个？每个窗口多装 1 个字节，
//        是为了让第 32 个字位（位置31）也有"正确答案"可猜——
//        我们曾经在这里栽过大跟头：那个位置从没被训练过，考试全瞎。
// 返回值：这批数据的平均损失（越小越好）
// ============================================================================
float forward(GPT& g, const unsigned char* toks, bool needLast = false) {
    // ---- 1. 查字典 + 对座位号：字节编号 -> 128 维向量 ----
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const float* er = &g.E[toks[b * (T + 1) + t] * D];   // 这个字的 embedding
            const float* pr = &g.P[(size_t)t * D];               // 这个座位的位置编码
            float* xr = &g.x[(b * T + t) * D];
            for (int d = 0; d < D; d++) xr[d] = er[d] + pr[d];   // 字义 + 位置，相加
        }
    // 注意：数据主通道 x 的布局是 16×32（每行 128 维），
    //       但读输入 toks 的步长是 33（每行 33 个字节）。两个"行宽"不一样，
    //       这正是"多装 1 个字"留下的痕迹 —— 【易错】改一处必须同时改另一处

    // ---- 2. 流过 4 层 Transformer（本课的主角）----
    static vector<float> attout;                 // 注意力支路的输出暂存区
    if (attout.empty()) attout.resize(BT * D);
    for (auto& l : g.layers) {

        // == 段1：多头注意力 ==
        copy(g.x.begin(), g.x.end(), l.x_in.begin());   // 存档原样（残差要用）
        layernorm_fwd(g.x.data(), l.ln1g.data(), l.ln1b.data(),
                      l.xhat1.data(), l.rstd1.data(), l.y1.data(), BT);   // 值日生整理
        // 用三对矩阵，把每个字投影成三种身份：
        gemm(l.y1.data(), l.Wq.data(), l.q.data(), BT, D, D); bias_add(l.q.data(), l.bq.data(), BT, D);  // 提问
        gemm(l.y1.data(), l.Wk.data(), l.k.data(), BT, D, D); bias_add(l.k.data(), l.bk.data(), BT, D);  // 挂牌
        gemm(l.y1.data(), l.Wv.data(), l.v.data(), BT, D, D); bias_add(l.v.data(), l.bv.data(), BT, D);  // 带料

        // 多头注意力（逐 batch、逐头，逻辑最清楚）
        // 【难】核心就一句话：每个字拿自己的 q，去和前面所有字的 k 算相似度，
        //        相似度变成权重，按权重把大家的 v 加权平均，得到"我该吸收的信息"。
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                float* att = &l.att[((size_t)b * H + h) * T * T];   // 分数表 32×32
                float* qh = &l.q[(size_t)b * T * D + h * NH];       // 本头的 q（起点偏移到本头那段）
                float* kh = &l.k[(size_t)b * T * D + h * NH];
                float* vh = &l.v[(size_t)b * T * D + h * NH];
                float* ch = &l.ctx[(size_t)b * T * D + h * NH];
                // 【记】qh 的基地址已经偏移到本头的起点，行内仍按 D=128 的
                //       整行宽度走（t*D），所以 qh[t*D+i] 恰好就是本头第 t 行
                //       第 i 个数——这不是 bug，是行优先排布的巧合之美
                for (int t = 0; t < T; t++) {
                    float mx = -1e30f, ssum = 0;
                    for (int j = 0; j <= t; j++) {      // 只看自己及前面的字（因果）
                        float s = 0;
                        for (int i = 0; i < NH; i++) s += qh[t * D + i] * kh[j * D + i];  // 点积=相似度
                        att[t * T + j] = s / SQRT_NH;   // 除以 √32，防数字过大
                        mx = max(mx, att[t * T + j]);
                    }
                    // softmax：把分数变成"加起来=1 的注意力权重"
                    // 【记】先减最大值再 exp，是防止 exp 爆掉（数值稳定小技巧）
                    for (int j = 0; j <= t; j++) { att[t * T + j] = expf(att[t * T + j] - mx); ssum += att[t * T + j]; }
                    for (int j = 0; j <= t; j++) att[t * T + j] /= ssum;
                    for (int j = t + 1; j < T; j++) att[t * T + j] = 0;   // 【易错】掩码补0：不许偷看未来的字
                }
                for (int t = 0; t < T; t++)
                    for (int i = 0; i < NH; i++) {
                        float s = 0;
                        for (int j = 0; j <= t; j++) s += att[t * T + j] * vh[j * D + i];   // 按注意力权重收料
                        ch[t * D + i] = s;
                    }
            }

        // 输出投影 + 残差：4 个头拼回的 128 维，再经 Wo 翻译一次
        gemm(l.ctx.data(), l.Wo.data(), attout.data(), BT, D, D);
        bias_add(attout.data(), l.bo.data(), BT, D);
        for (int i = 0; i < BT * D; i++) g.x[i] += attout[i];   // 残差：加回原样

        // == 段2：前馈网络（FFN）==
        copy(g.x.begin(), g.x.end(), l.x_in2.begin());      // 再次存档原样
        layernorm_fwd(g.x.data(), l.ln2g.data(), l.ln2b.data(),
                      l.xhat2.data(), l.rstd2.data(), l.y2.data(), BT);   // 值日生整理
        gemm(l.y2.data(), l.W1.data(), l.hid.data(), BT, D, F);   // 128 -> 512 摊开
        bias_add(l.hid.data(), l.b1.data(), BT, F);
        for (auto& h_ : l.hid) h_ = max(h_, 0.0f);      // ReLU：负数清零（只留下有用的特征）
        gemm(l.hid.data(), l.W2.data(), l.ffn.data(), BT, F, D);   // 512 -> 128 提炼
        bias_add(l.ffn.data(), l.b2.data(), BT, D);
        for (int i = 0; i < BT * D; i++) g.x[i] += l.ffn[i];   // 残差：再加回原样
    }

    // ---- 3. 收尾：顶层整理 -> 打分 -> softmax -> 算损失 ----
    layernorm_fwd(g.x.data(), g.lnFg.data(), g.lnFb.data(),
                  g.xhatf.data(), g.rstdf.data(), g.yf.data(), BT);
    gemm(g.yf.data(), g.WU.data(), g.probs.data(), BT, D, V);  // 128 -> 256 个原始分数(logits)
    int tlim = T;   // 【易错】最后一列也监督，必须 softmax，不能省！
                    // （我们曾经为了省一点点时间跳过最后一列，结果那一列
                    //   拿着没归一化的 logits 当概率算损失，梯度全错）
    for (int b = 0; b < B; b++)
        for (int t = 0; t < tlim; t++) {
            float* row = &g.probs[((size_t)b * T + t) * V];
            float mx = -1e30f, s = 0;
            for (int y = 0; y < V; y++) mx = max(mx, row[y]);   // 老规矩：先减最大值
            for (int y = 0; y < V; y++) { row[y] = expf(row[y] - mx); s += row[y]; }
            for (int y = 0; y < V; y++) row[y] /= s;            // 现在是真正的概率了
        }

    // 交叉熵损失：正确答案的"惊讶程度" = -log(正确答案的概率)
    // 【记】概率越接近 1，损失越接近 0；瞎猜时 256 类各 1/256，损失 = ln256 ≈ 5.545
    float loss = 0; int cnt = 0;
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            int target = toks[b * (T + 1) + t + 1];   // 正确答案：窗口里的下一个字节
            loss += -logf(max(g.probs[((size_t)b * T + t) * V + target], 1e-12f));
            cnt++;
        }
    return loss / cnt;
}

// ============================================================================
// 第6区：反向传播 —— 错在哪，一层层传回去
// ----------------------------------------------------------------------------
// 总思想（第2、3课已经打过基础）：
//   1. 从损失出发，算出每个输出概率的"责任"：(概率 - 正确答案)
//   2. 用链式法则从后往前：每遇到一个运算，就问"输出变了，输入该改多少"
//   3. 每过一个矩阵，顺便把矩阵自己的梯度也攒出来
// 阅读顺序：先认出"这是第5区的倒放"，再逐行对号入座。
// ============================================================================
void backward(GPT& g, const unsigned char* toks) {
    static vector<float> dx, dy, tmp, dq, dk, dv, dctx2, datt, dffn, dlogits;
    if (dx.empty()) {   // 这些缓冲区只申请一次（static），反复使用
        dx.resize(BT * D); dy.resize(BT * D); tmp.resize(BT * D);
        dq.resize(BT * D); dk.resize(BT * D); dv.resize(BT * D); dctx2.resize(BT * D);
        datt.resize(T * T); dffn.resize(BT * D);
        dlogits.resize((size_t)BT * V);
    }
    const int NVALID = B * T;   // 本批有效预测数 = 16×32 = 512
                                // 【记】梯度除以样本数 = 求平均，步子不会随批量忽大忽小
    fill(dx.begin(), dx.end(), 0.0f);
    fill(dlogits.begin(), dlogits.end(), 0.0f);

    // -- 1. softmax+交叉熵 的梯度 = (概率 - 正确答案) / 样本数 --
    // 【记】这是全世界最优雅的公式之一：猜对了，概率-1≈0，不用改；
    //        猜错了，差额就是"该往正确答案方向推多少"
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            float* row = &dlogits[((size_t)b * T + t) * V];
            const float* pr = &g.probs[((size_t)b * T + t) * V];
            int target = toks[b * (T + 1) + t + 1];
            for (int y = 0; y < V; y++)
                row[y] = (pr[y] - (y == target ? 1.0f : 0.0f)) / NVALID;
        }

    // -- 2. 输出头 WU：反向穿过"128 -> 256"的打分矩阵 --
    gemm_at(g.yf.data(), dlogits.data(), g.g_WU.data(), D, BT, V);   // WU 的梯度
    gemm_bt(dlogits.data(), g.WU.data(), dx.data(), BT, V, D);       // 回传给 yf 的梯度

    // -- 3. 顶层 LayerNorm 反传 --
    layernorm_bwd(g.xhatf.data(), g.rstdf.data(), g.lnFg.data(), dx.data(), dy.data(),
                  g.g_lnFg.data(), g.g_lnFb.data(), BT);
    copy(dy.begin(), dy.end(), dx.begin());   // dx 现在代表"顶层 LN 输入"的梯度

    // -- 4. 逐层反传（倒序！最后学的层最先改）--
    for (int li = L - 1; li >= 0; li--) {
        auto& l = g.layers[li];

        // --- 段2（FFN）支路，按前向的倒序拆 ---
        copy(dx.begin(), dx.end(), dffn.begin());           // 残差岔口：梯度兵分两路
        colsum(dffn.data(), l.g_b2.data(), BT, D);                       // b2 的梯度
        gemm_at(l.hid.data(), dffn.data(), l.g_W2.data(), F, BT, D);     // W2 的梯度
        gemm_bt(dffn.data(), l.W2.data(), tmp.data(), BT, D, F);   // 回传到 512 维层
        for (int i = 0; i < BT * F; i++) if (l.hid[i] == 0) tmp[i] = 0;   // ReLU反传：
            // 【记】前向时被 ReLU 清零的神经元，反向也没有梯度（死掉的通道不背锅）
        colsum(tmp.data(), l.g_b1.data(), BT, F);                        // b1 的梯度
        gemm_at(l.y2.data(), tmp.data(), l.g_W1.data(), D, BT, F);       // W1 的梯度
        gemm_bt(tmp.data(), l.W1.data(), dy.data(), BT, F, D);     // 回传到 y2
        layernorm_bwd(l.xhat2.data(), l.rstd2.data(), l.ln2g.data(), dy.data(),
                      tmp.data(), l.g_ln2g.data(), l.g_ln2b.data(), BT);   // LN2 反传
        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];   // 残差另一路的梯度汇回来

        // --- 段1（注意力）支路 ---
        colsum(dx.data(), l.g_bo.data(), BT, D);                    // bo 的梯度
        gemm_at(l.ctx.data(), dx.data(), l.g_Wo.data(), D, BT, D);  // Wo 的梯度
        gemm_bt(dx.data(), l.Wo.data(), dctx2.data(), BT, D, D);    // 回传到各头的"收货单"

        // 多头注意力的反向：把上面那份"收货单"按注意力权重退回给每个字
        fill(dq.begin(), dq.end(), 0.0f);
        fill(dk.begin(), dk.end(), 0.0f);
        fill(dv.begin(), dv.end(), 0.0f);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                const float* att = &l.att[((size_t)b * H + h) * T * T];   // 前向存的权重表
                const float* qh = &l.q[(size_t)b * T * D + h * NH];
                const float* kh = &l.k[(size_t)b * T * D + h * NH];
                const float* vh = &l.v[(size_t)b * T * D + h * NH];
                float* cq = &dq[(size_t)b * T * D + h * NH];   // 本头 q 的梯度
                float* ck = &dk[(size_t)b * T * D + h * NH];
                float* cv = &dv[(size_t)b * T * D + h * NH];
                // dctx2 里取本头第 t 行第 i 个数的快捷方式
                auto dcx = [&](int t, int i) {
                    return dctx2[((size_t)b * T + t) * D + h * NH + i];
                };
                // dv[j] = Σ_{t≥j} att[t,j] · dctx[t]
                // 读法：字 j 的 v 被所有"看过它"的字 t 按权重退回了梯度
                for (int j = 0; j < T; j++)
                    for (int i = 0; i < NH; i++) {
                        float s = 0;
                        for (int t = j; t < T; t++) s += att[t * T + j] * dcx(t, i);
                        cv[j * D + i] = s;
                    }
                // dscore（softmax 反传）
                // 【难】softmax 的反向公式：datt[t,j] = att[t,j] × (s[t,j] − 行内加权和)
                //        意思是：每个分数的梯度，要扣掉"全行平均意见"，
                //        因为 softmax 里所有分数互相竞争、此消彼长
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
                // 相似度 = q·k，所以梯度一个给 q、一个给 k：
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
        // 三个矩阵的梯度：谁参与乘法，谁攒一份（这就是链式法则的"分账"）
        gemm_bt(dq.data(), l.Wq.data(), dy.data(), BT, D, D);
        gemm_bt(dk.data(), l.Wk.data(), tmp.data(), BT, D, D);
        for (int i = 0; i < BT * D; i++) dy[i] += tmp[i];
        gemm_bt(dv.data(), l.Wv.data(), tmp.data(), BT, D, D);
        for (int i = 0; i < BT * D; i++) dy[i] += tmp[i];
        gemm_at(l.y1.data(), dq.data(), l.g_Wq.data(), D, BT, D); colsum(dq.data(), l.g_bq.data(), BT, D);
        gemm_at(l.y1.data(), dk.data(), l.g_Wk.data(), D, BT, D); colsum(dk.data(), l.g_bk.data(), BT, D);
        gemm_at(l.y1.data(), dv.data(), l.g_Wv.data(), D, BT, D); colsum(dv.data(), l.g_bv.data(), BT, D);
        // LN1 反传 + 残差
        layernorm_bwd(l.xhat1.data(), l.rstd1.data(), l.ln1g.data(), dy.data(),
                      tmp.data(), l.g_ln1g.data(), l.g_ln1b.data(), BT);
        for (int i = 0; i < BT * D; i++) dx[i] += tmp[i];
    }

    // -- 5. 最源头：嵌入表和位置表的梯度 --
    // 哪个字用了哪行 embedding，就把梯度累加回哪行（同一个字可能用多次）
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const float* xr = &dx[((size_t)b * T + t) * D];
            float* er = &g.g_E[toks[b * (T + 1) + t] * D];
            float* pr = &g.g_P[(size_t)t * D];
            for (int d = 0; d < D; d++) { er[d] += xr[d]; pr[d] += xr[d]; }
        }
}

// ============================================================================
// 第7区：Adam —— 带惯性的改错本
// ----------------------------------------------------------------------------
// 【难】Adam = 梯度下降（第2课）的豪华升级版，两个记忆：
//   m（动量）：梯度的"加权历史平均"——像惯性，方向一致的梯度会越攒越大，
//             山路抖动被抹平（beta1 = 0.9，即平均最近约 10 步）
//   v（二阶矩）：梯度平方的历史平均——记录每条路 historically 有多陡，
//             陡的路自动小步走、缓的路大步走（beta2 = 0.999）
//   c1/c2： bias correction（偏差修正）。m、v 从 0 起步，开头几步天然偏小，
//          除以 (1 - beta^t) 把它放大回正常尺度。【记】没有它，开局会踉跄
// ============================================================================
struct Adam {
    unordered_map<vector<float>*, vector<float>> m1, m2;
    // 【记】用参数的内存地址当 key，每个参数向量配一对 m/v —— 零额外登记
    void step(vector<float>& p, vector<float>& g, int t, float lr) {
        auto& M = m1[&p]; auto& Vv = m2[&p];
        if (M.empty()) { M.assign(p.size(), 0); Vv.assign(p.size(), 0); }
        float c1 = 1 - powf(0.9f, t), c2 = 1 - powf(0.999f, t);
        for (size_t i = 0; i < p.size(); i++) {
            M[i] = 0.9f * M[i] + 0.1f * g[i];        // 更新动量
            Vv[i] = 0.999f * Vv[i] + 0.001f * g[i] * g[i];   // 更新二阶矩
            p[i] -= lr * (M[i] / c1) / (sqrtf(Vv[i] / c2) + 1e-8f);
            //        ^^ 学习率  ^^ 修正后的动量  ^^ 除以"历史陡度"，加 1e-8 防除 0
        }
    }
};
void applyAdam(GPT& g, Adam& a, int t, float lr) {   // 把全模型 22 组参数逐个交给 Adam
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

// ============================================================================
// 第8区：主流程 —— 读语料 -> 训练循环 -> 期末考/仿写
// ----------------------------------------------------------------------------
// 命令行三个参数（都可省略）：
//   lesson4_gpt.exe [训练步数] [读取存档] [保存存档]
//   例： lesson4_gpt.exe 800 gpt4_c.bin gpt4_c.bin   （从存档续训 800 步）
//        lesson4_gpt.exe 0 gpt4_c.bin gpt4_c.bin     （步数=0：不训练，只考试+仿写）
// 【易错】"没有存档"要传占位符 none —— 传空字符串 "" 会让程序启动失败（exit 127）
// ============================================================================
int main(int argc, char** argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 100;                 // 训练步数
    const char* loadfile = argc > 2 ? argv[2] : nullptr;        // 开局读哪个存档（可空）
    const char* savefile = argc > 3 ? argv[3] : "gpt4.bin";     // 结束存到哪里

    // ---- 读语料：179MB 的六大风格混合语料，一次性全读进内存 ----
    ifstream fin("all_corpus.bin", ios::binary);
    vector<unsigned char> data((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();
    if (data.size() < 1000000) {   // 【易错】文件打不开时给个明白话，不要神秘崩溃
        cout << "打不开 all_corpus.bin！请在 cpp 目录下运行" << endl;
        return 1;
    }
    cout << "语料: all_corpus.bin " << data.size() / 1000000 << " MB" << endl;

    mt19937 rng(42);        // 随机数种子固定为 42：每次"随机"都一样，实验可复现
    GPT g;
    initGPT(g, rng);        // 全部参数从小随机数起步
    cout << "参数量: " << g.nparams() / 1000 << " K" << endl;

    // ---- 如果指定了存档，把训练好的参数读回来，覆盖刚初始化的随机值 ----
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

    // ================= 评测模式（步数 = 0）：考卷 + 仿写 =================
    if (steps == 0) {
        // 考卷：从文本开头 sliding window（滑动窗口）逐个位置考 3000 道题
        auto exam = [&](const char* name, const unsigned char* d, size_t n) {
            double loss = 0; long long hit = 0, cnt = 0;
            for (size_t pos = 0; pos + T < n && cnt < 3000; pos++) {
                static vector<unsigned char> w;
                if (w.empty()) w.resize(B * (T + 1));   // 只装第 0 个窗口的 32 字节
                for (int t = 0; t < T; t++) w[t] = d[pos + t];
                forward(g, w.data(), true);
                const float* pr = &g.probs[(T - 1) * V];   // 只考窗口最后一个位置：
                int target = d[pos + T];                   // 看满 32 字节后猜下一个
                loss += -logf(max(pr[target], 1e-12f));
                int best = 0;                              // 模型最看好的字
                for (int y = 1; y < V; y++) if (pr[y] > pr[best]) best = y;
                if (best == target) hit++;
                cnt++;
            }
            printf("考卷[%s] 平均损失 %.3f  准确率 %.1f%%\n", name, loss / cnt, 100.0 * hit / cnt);
        };
        // 三张考卷：红楼梦原文 + 三体 + 天工切片（每个只考前 6 万字节）
        ifstream f1("hongloumeng.txt", ios::binary);
        vector<unsigned char> d1((istreambuf_iterator<char>(f1)), istreambuf_iterator<char>());
        exam("红楼梦·古文", d1.data(), 60000);
        ifstream f2("santi1.txt", ios::binary);
        vector<unsigned char> d2((istreambuf_iterator<char>(f2)), istreambuf_iterator<char>());
        exam("三体·科幻", d2.data(), 60000);
        ifstream f3("zh_modern.txt", ios::binary);
        vector<unsigned char> d3((istreambuf_iterator<char>(f3)), istreambuf_iterator<char>());
        exam("天工·现代文", d3.data(), 60000);

        // 仿写：在语料里找到"笑道"两个字（UTF-8 字节序列 231,172,145,233,129,147），
        //       从那里开始让模型一个字一个字续写 500 字节
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
            // 按概率随机抽样（温度 0.7 = 把概率开 1/0.7 次方后再归一化：
            // 高温更敢乱猜，低温总是选最大。0.7 让文章有一点惊喜但不失控）
            double q[256], s = 0;
            for (int y = 0; y < 256; y++) { q[y] = pow(pr[y], 1.0 / 0.7); s += q[y]; }
            double r = uniform_real_distribution<double>(0, 1)(rng) * s;
            int nxt = 0;
            while (r > q[nxt]) { r -= q[nxt]; nxt++; }   // 轮盘赌抽样
            out.push_back((char)w[0]);                   // 窗口最旧的字"落地"
            for (int t = 0; t < T - 1; t++) w[t] = w[t + 1];   // 整体左移一格
            w[T - 1] = (unsigned char)nxt;               // 新字进窗口末尾
        }
        out.append((char*)&w[0], T);                     // 把窗口里剩下的字也倒出来
        ofstream fout("gen4.bin", ios::binary);
        fout.write(out.data(), out.size());
        fout.close();
        cout << "仿写已存 gen4.bin（500字节）" << endl;
        return 0;
    }

    // ================= 训练模式：核心循环（每一步三件事）=================
    Adam adam;
    vector<unsigned char> batch(B * (T + 1));   // 16 个窗口 × 每窗 33 字节
    double runLoss = 0; int runN = 0;
    time_t t0 = time(nullptr);
    for (int step = 1; step <= steps; step++) {
        // ① 抓数据：在语料里随机挖 16 段连续的 33 字节
        for (int b = 0; b < B; b++) {
            size_t pos = rng() % (data.size() - T);
            for (int t = 0; t <= T; t++) batch[b * (T + 1) + t] = data[pos + t];
        }
        // ② 清零上一轮的意见书，重新前向+反向
        for (auto& l : g.layers) l.zeroGrad();
        fill(g.g_E.begin(), g.g_E.end(), 0.0f);
        fill(g.g_P.begin(), g.g_P.end(), 0.0f);
        fill(g.g_lnFg.begin(), g.g_lnFg.end(), 0.0f);
        fill(g.g_lnFb.begin(), g.g_lnFb.end(), 0.0f);
        fill(g.g_WU.begin(), g.g_WU.end(), 0.0f);

        float loss = forward(g, batch.data());   // 前向：算损失
        backward(g, batch.data());               // 反向：攒出每个参数的梯度
        float lr = step < steps * 0.6 ? 0.001f : 0.0003f;   // 先大后小（学习率衰减）
        applyAdam(g, adam, step, lr);            // ③ Adam 按梯度修改所有参数

        // 每 20 步汇报一次：损失和耗时（随机水平 = ln256 ≈ 5.545 是参照线）
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
