// ============================================================
// C++ 造大语言模型 · 第2课附加：模型期末考
// ------------------------------------------------------------
// 用四大名著训练 bigram，然后分别在三份考卷上测"平均损失 + 预测准确率"：
//   考卷A: 红楼梦     —— 学过的内容（开卷考）
//   考卷B: 天工现代文 —— 没学过 + 时代不同（闭卷跨时代考）
//   考卷C: 三体       —— 没学过 + 科幻腔（闭卷跨风格考）
// 顺便把模型写的东西存盘，交给 Python 检查 UTF-8 合格率。
// ============================================================
#include <bits/stdc++.h>
using namespace std;

vector<unsigned char> readFile(const char* fn) {
    ifstream fin(fn, ios::binary);
    vector<unsigned char> d((istreambuf_iterator<char>(fin)),
                            istreambuf_iterator<char>());
    fin.close();
    return d;
}

int main() {
    // ---------- 训练（和第2课完全相同，保证可复现） ----------
    const char* files[] = {"hongloumeng.txt", "sanguoyanyi.txt",
                           "shuihuzhuan.txt", "xiyouji.txt"};
    vector<unsigned char> data;
    for (auto fn : files) {
        auto part = readFile(fn);
        data.insert(data.end(), part.begin(), part.end());
    }
    cout << "训练语料: 四大名著 " << data.size() << " 字节" << endl;

    vector<vector<double>> W(256, vector<double>(256, 0.0));
    mt19937 rng(42);
    for (int step = 1; step <= 300000; step++) {
        int pos = rng() % (data.size() - 1);
        int a = data[pos], b = data[pos + 1];
        double mx = *max_element(W[a].begin(), W[a].end());
        double sum = 0, p[256];
        for (int y = 0; y < 256; y++) { p[y] = exp(W[a][y] - mx); sum += p[y]; }
        for (int y = 0; y < 256; y++) p[y] /= sum;
        for (int y = 0; y < 256; y++)
            W[a][y] -= 0.1 * (p[y] - (y == b ? 1.0 : 0.0));
    }
    cout << "训练完成" << endl;

    // ---------- 考试函数：平均损失 + top1 准确率 ----------
    auto exam = [&](const char* name, vector<unsigned char> t) {
        if (t.size() > 1000000) t.resize(1000000);   // 每份考卷只考前 100 万字节
        double loss = 0; long long hit = 0;
        for (size_t i = 0; i + 1 < t.size(); i++) {
            int a = t[i], b = t[i + 1];
            double mx = *max_element(W[a].begin(), W[a].end());
            double sum = 0, p[256];
            for (int y = 0; y < 256; y++) { p[y] = exp(W[a][y] - mx); sum += p[y]; }
            for (int y = 0; y < 256; y++) p[y] /= sum;
            loss += -log(p[b]);
            int best = 0;                       // 模型最看好的字节
            for (int y = 1; y < 256; y++) if (p[y] > p[best]) best = y;
            if (best == b) hit++;
        }
        printf("考卷[%s] 平均损失 %.3f  预测准确率 %.1f%%  (%zu 字节)\n",
               name, loss / (t.size() - 1),
               100.0 * hit / (t.size() - 1), t.size());
    };

    exam("红楼梦·学过", readFile("hongloumeng.txt"));
    exam("天工·现代文·没学过", readFile("zh_modern.txt"));
    exam("三体·科幻·没学过", readFile("santi1.txt"));

    // ---------- 写一段模型仿写，存盘给 Python 检查 ----------
    int cur = 233;   // 从"道"字第一字节开头，让它写段正经的
    string out;
    for (int i = 0; i < 600; i++) {
        double m2 = *max_element(W[cur].begin(), W[cur].end());
        double s2 = 0, q[256];
        for (int y = 0; y < 256; y++) { q[y] = exp(W[cur][y] - m2); s2 += q[y]; }
        double r = uniform_real_distribution<>(0, 1)(rng) * s2;
        int nxt = 0;
        while (r > q[nxt]) { r -= q[nxt]; nxt++; }
        out.push_back((char)cur);
        cur = nxt;
    }
    out.push_back((char)cur);
    ofstream fout("sample.bin", ios::binary);
    fout.write(out.data(), out.size());
    fout.close();
    cout << "模型仿写已存到 sample.bin（600 字节）" << endl;
    return 0;
}
