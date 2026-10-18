// ============================================================
// C++ 造大语言模型 · 第2课：bigram 神经网络
// ------------------------------------------------------------
// 第1课我们数出了"每个字节出现多少次"（死记硬背）。
// 这一课让模型学会预测："看到字节 a，下一个字节最可能是谁？"
//
// 模型 = 一张 256×256 的分数表 W。
//   W[a][b] = 看完字节 a 之后，下一个字节是 b 的"原始分数"
//
// 训练循环（每步只做四件事）：
//   1. 随机抓一对相邻字节 (a, b)        —— a 是题目，b 是标准答案
//   2. 把 W[a] 这一行 softmax 成概率    —— 模型自己的猜测
//   3. 交叉熵：-log(猜测在标准答案上的概率) —— 算错得多离谱
//   4. 梯度下降：把 W[a] 这一行往标准答案方向推一小步
// 走 30 万步，整张表就被"雕刻"出来了。
// ============================================================
// 标注说明：【难】=重点看注释 【易错】=容易写错 【记】=照抄即可
// ============================================================
#include <bits/stdc++.h>   // 万能头
using namespace std;

int main() {
    // ---------- 1. 读入四大名著，拼成一大串字节 ----------
    const char* files[] = {"hongloumeng.txt", "sanguoyanyi.txt",
                           "shuihuzhuan.txt", "xiyouji.txt"};
    vector<unsigned char> data;
    for (auto fn : files) {
        ifstream fin(fn, ios::binary);          // 【易错】ios::binary
        if (!fin) { cout << "找不到 " << fn << endl; return 1; }
        vector<unsigned char> part(
            (istreambuf_iterator<char>(fin)),
            istreambuf_iterator<char>());
        data.insert(data.end(), part.begin(), part.end());
        fin.close();
    }
    cout << "训练语料: 四大名著，共 " << data.size() << " 个字节" << endl;

    // ---------- 2. 模型本体：256×256 分数表 ----------
    // W[a][b] = 看到 a 后，下一个字节是 b 的分数
    vector<vector<double>> W(256, vector<double>(256, 0.0));
    // 【记】softmax 回归从全 0 开始就能学，不需要随机初始化

    // ---------- 3. 训练 ----------
    mt19937 rng(42);                 // 【记】随机数发生器，42 是种子，保证每次结果一样
    const int STEPS = 300000;        // 训练步数
    const double LR = 0.1;           // 学习率：每步推多大一步

    double lossSum = 0;              // 累计损失，用来打印"最近错得有多离谱"
    int lossCnt = 0;

    for (int step = 1; step <= STEPS; step++) {
        // (1) 随机抓一对相邻字节
        int pos = rng() % (data.size() - 1);
        int a = data[pos];           // 题目
        int b = data[pos + 1];       // 标准答案

        // (2) softmax：把 W[a] 这一行 256 个分数转成 256 个概率
        // 【难】softmax = 先取最大值防止指数爆炸，再 exp，再归一化
        double mx = *max_element(W[a].begin(), W[a].end());
        double sum = 0;
        double p[256];
        for (int y = 0; y < 256; y++) {
            p[y] = exp(W[a][y] - mx);   // 【易错】要先减 mx，否则 exp 溢出
            sum += p[y];
        }
        for (int y = 0; y < 256; y++) p[y] /= sum;

        // (3) 交叉熵损失 = -log(模型给标准答案 b 的概率)
        lossSum += -log(p[b]);
        lossCnt++;

        // (4) 梯度下降：softmax+交叉熵的梯度恰好是 (概率 - 正确答案)
        // 【难】推导过程你可以先记住结论：
        //   对 dLoss/dW[a][y] = p[y] - (y == b ? 1 : 0)
        //   意思是：概率给高了就压下去，给低了就抬上来，其他 255 格微降。
        for (int y = 0; y < 256; y++) {
            double grad = p[y] - (y == b ? 1.0 : 0.0);
            W[a][y] -= LR * grad;    // 往"错得少"的方向推一小步
        }

        // 每 5 万步报告一次平均损失
        if (step % 50000 == 0) {
            cout << "step " << step << "  平均损失 = "
                 << fixed << setprecision(3) << lossSum / lossCnt
                 << "  （随机瞎猜是 " << log(256.0) << "）" << endl;
            lossSum = 0; lossCnt = 0;
        }
    }

    // ---------- 4. 抽查：模型觉得"道"字第一字节后面跟什么？ ----------
    // "道"的 UTF-8 = 233, 129, 147。给它看 233，它应该强烈预测 129。
    int test = 233;
    double mx = *max_element(W[test].begin(), W[test].end());
    double sum = 0, p[256];
    for (int y = 0; y < 256; y++) { p[y] = exp(W[test][y] - mx); sum += p[y]; }
    for (int y = 0; y < 256; y++) p[y] /= sum;

    vector<int> ord(256);
    for (int i = 0; i < 256; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int x, int y){ return p[x] > p[y]; });

    cout << "看到字节 233（汉字'道'的第 1/3 字节），模型预测下一个是：" << endl;
    for (int k = 0; k < 3; k++)
        printf("  字节 %3d  概率 %.4f\n", ord[k], p[ord[k]]);

    // ---------- 5. 让模型自己写一段《红楼梦》----------
    // 从一个随机位置的字节出发，按模型给出的概率一步步采样 400 次
    int cur = data[rng() % data.size()];
    string out;
    for (int i = 0; i < 400; i++) {
        // 当前字节的概率分布
        double m2 = *max_element(W[cur].begin(), W[cur].end());
        double s2 = 0, q[256];
        for (int y = 0; y < 256; y++) { q[y] = exp(W[cur][y] - m2); s2 += q[y]; }
        // 【难】按概率采样：扔一个 [0,1) 的随机数，落在哪个概率区间就选谁
        double r = uniform_real_distribution<>(0, 1)(rng) * s2;
        int nxt = 0;
        while (r > q[nxt]) { r -= q[nxt]; nxt++; }
        out.push_back((char)cur);   // 记录当前字节
        cur = nxt;                  // 把采样结果当作下一个题目
    }
    out.push_back((char)cur);
    cout << endl << "模型仿写的《红楼梦》片段（400 字节）：" << endl;
    cout << out << endl;
    return 0;
}
