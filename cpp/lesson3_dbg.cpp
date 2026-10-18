// 调试版：监控每一步、每个参数组的最大梯度
#include <bits/stdc++.h>
using namespace std;

const int T = 8, D = 16;

struct Model {
    vector<vector<double>> E, Wq, Wk, Wv, U;
};

Model initModel(mt19937& rng) {
    Model m;
    auto rnd = [&](int r, int c) {
        vector<vector<double>> v(r, vector<double>(c));
        uniform_real_distribution<double> u(-0.02, 0.02);
        for (auto& row : v) for (auto& x : row) x = u(rng);
        return v;
    };
    m.E = rnd(256, D); m.Wq = rnd(D, D); m.Wk = rnd(D, D);
    m.Wv = rnd(D, D); m.U = rnd(D, 256);
    return m;
}

double forward(const Model& m, const unsigned char* ctx, int target) {
    double x[T][D];
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) x[i][d] = m.E[ctx[i]][d];
    double Q[T][D], K[T][D], V[T][D];
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) {
            Q[i][d] = K[i][d] = V[i][d] = 0;
            for (int e = 0; e < D; e++) {
                Q[i][d] += x[i][e] * m.Wq[e][d];
                K[i][d] += x[i][e] * m.Wk[e][d];
                V[i][d] += x[i][e] * m.Wv[e][d];
            }
        }
    double A[T][T];
    for (int i = 0; i < T; i++) {
        double mx = -1e18, sum = 0;
        for (int j = 0; j <= i; j++) {
            double s = 0;
            for (int d = 0; d < D; d++) s += Q[i][d] * K[j][d];
            A[i][j] = s / sqrt((double)D);
            mx = max(mx, A[i][j]);
        }
        for (int j = 0; j <= i; j++) { A[i][j] = exp(A[i][j] - mx); sum += A[i][j]; }
        for (int j = 0; j <= i; j++) A[i][j] /= sum;
    }
    double o[D] = {0};
    for (int j = 0; j < T; j++)
        for (int d = 0; d < D; d++) o[d] += A[T-1][j] * V[j][d];
    double logits[256], mx = -1e18, sum = 0, p[256];
    for (int y = 0; y < 256; y++) {
        logits[y] = 0;
        for (int d = 0; d < D; d++) logits[y] += o[d] * m.U[d][y];
        mx = max(mx, logits[y]);
    }
    for (int y = 0; y < 256; y++) { p[y] = exp(logits[y] - mx); sum += p[y]; }
    for (int y = 0; y < 256; y++) p[y] /= sum;
    return -log(p[target]);
}

int main() {
    ifstream fin("hongloumeng.txt", ios::binary);
    vector<unsigned char> data((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();
    mt19937 rng(42);
    Model m = initModel(rng);
    const double LR = 0.01, EPS = 1e-6;
    const int STEPS = 3;

    for (int step = 1; step <= STEPS; step++) {
        int pos[4]; unsigned char ctx[4][T]; int tgt[4];
        for (int s = 0; s < 4; s++) {
            pos[s] = T + rng() % (data.size() - T - 1);
            for (int i = 0; i < T; i++) ctx[s][i] = data[pos[s] - T + i];
            tgt[s] = data[pos[s]];
        }
        auto batchLoss = [&](Model& mm) {
            double l = 0;
            for (int s = 0; s < 4; s++) l += forward(mm, ctx[s], tgt[s]);
            return l / 4;
        };
        double base = batchLoss(m);
        cout << "step " << step << " base = " << base << endl;
        for (int s = 0; s < 4; s++)
            cout << "   样本" << s << " 损失 = " << forward(m, ctx[s], tgt[s])
                 << "  目标字节 = " << tgt[s] << endl;

        vector<int> used;
        for (int s = 0; s < 4; s++) for (int i = 0; i < T; i++) used.push_back(ctx[s][i]);
        sort(used.begin(), used.end());
        used.erase(unique(used.begin(), used.end()), used.end());

        double gmaxE = 0, gmaxW = 0, gmaxU = 0;
        auto nudge = [&](vector<vector<double>>& M, int r, int c, double& gmax) {
            M[r][c] += EPS;
            double l1 = batchLoss(m);
            double g = (l1 - base) / EPS;
            if (fabs(g) > gmax) gmax = fabs(g);
            if (fabs(g) > 1000)
                cout << "    !! 巨型梯度 g=" << g << " at [" << r << "][" << c << "] l1=" << l1 << endl;
            M[r][c] -= EPS;
            M[r][c] -= LR * g;
        };
        for (int r : used) for (int d = 0; d < D; d++) nudge(m.E, r, d, gmaxE);
        for (int a = 0; a < D; a++) for (int b = 0; b < D; b++) {
            nudge(m.Wq, a, b, gmaxW); nudge(m.Wk, a, b, gmaxW); nudge(m.Wv, a, b, gmaxW);
        }
        for (int d = 0; d < D; d++) for (int y = 0; y < 256; y++) nudge(m.U, d, y, gmaxU);
        cout << "   max|g| E=" << gmaxE << " W=" << gmaxW << " U=" << gmaxU << endl;
    }
    return 0;
}
