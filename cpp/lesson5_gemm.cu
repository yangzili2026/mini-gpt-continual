// ============================================================
// C++ 造大语言模型 · 第5课（上）：第一个 CUDA kernel —— 矩阵乘法
// ------------------------------------------------------------
// CPU 算矩阵乘是一个数一个数地循环；
// GPU 的想法：矩阵有 m×n 个"答案格子"，让几万个线程同时填格子。
// 本课三件事：
//   1. naive kernel   ：一个线程填一个格子（最好懂，浪费带宽）
//   2. tiled kernel   ：把矩阵切成 16×16 瓷砖，共享内存复用（【难】但快）
//   3. 和 CPU 对答案 + 跑分
// 编译（CUDA 13.3 + VS2026）：
//   nvcc -O3 -arch=sm_89 -ccbin "<VS2026 Hostx64 x64 目录>" -o lesson5_gemm.exe lesson5_gemm.cu
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda_runtime.h>

// ---------------- CPU 版（和第4课同款） ----------------
void gemm_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {
    memset(C, 0, sizeof(float) * (size_t)m * n);
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++) {
            float a = A[i * k + l];
            const float* Brow = Bm + (size_t)l * n;
            float* Crow = C + (size_t)i * n;
            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
}

// ---------------- GPU 版 1：朴素 kernel ----------------
// 一个线程负责 C 的一个格子 (i, j)：把这个格子的一行乘一列加起来。
__global__ void gemm_naive(const float* A, const float* Bm, float* C, int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // 第几行
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // 第几列
    if (i >= m || j >= n) return;
    float s = 0;
    for (int l = 0; l < k; l++) s += A[i * k + l] * Bm[l * n + j];
    C[i * n + j] = s;
}

// ---------------- GPU 版 2：瓷砖 kernel（【难】核心思想） ----------------
// 朴素版慢的原因：每算一个格子，都要把 A 的一行、B 的一列从显存重新搬一遍。
// 瓷砖版：一个线程块(block)负责 C 的 16×16 一块；
//   先把要用的 A 条带和 B 条带搬进"共享内存"（块内所有线程都能用的高速小仓库），
//   16×16 个格子合起来只搬一遍，显存流量省 16 倍。
#define TS 16   // 瓷砖边长（tile size）
__global__ void gemm_tiled(const float* A, const float* Bm, float* C, int m, int k, int n) {
    __shared__ float sa[TS][TS];   // 【易错】shared 内存：块内共享，出了块就没了
    __shared__ float sb[TS][TS];

    int tx = threadIdx.x, ty = threadIdx.y;        // 线程在瓷砖内的坐标
    int i = blockIdx.y * TS + ty;                  // 全局行
    int j = blockIdx.x * TS + tx;                  // 全局列

    float s = 0;
    // 沿 k 方向一块一块搬：每循环一次处理 k 方向的 TS 个
    for (int k0 = 0; k0 < k; k0 += TS) {
        // 各线程各搬一个数进共享内存（越界补 0，免得把垃圾乘进去）
        sa[ty][tx] = (i < m && k0 + tx < k) ? A[i * k + k0 + tx] : 0.0f;
        sb[ty][tx] = (k0 + ty < k && j < n) ? Bm[(k0 + ty) * n + j] : 0.0f;
        __syncthreads();                           // 【易错】等全块 256 个线程都搬完再算
        for (int l = 0; l < TS; l++) s += sa[ty][l] * sb[l][tx];
        __syncthreads();                           // 【易错】下一轮搬运前，先用完本轮的
    }
    if (i < m && j < n) C[i * n + j] = s;
}

// ---------------- 对答案工具 ----------------
float max_diff(const vector<float>& x, const vector<float>& y) {
    float d = 0;
    for (size_t i = 0; i < x.size(); i++) d = max(d, fabsf(x[i] - y[i]));
    return d;
}

int main() {
    // 1) 小矩阵对答案
    int m = 512, k = 384, n = 256;
    mt19937 rng(42);
    uniform_real_distribution<float> ud(-1.0f, 1.0f);
    vector<float> A(m * k), Bm(k * n), C0(m * n), C1(m * n);
    for (auto& x : A) x = ud(rng);
    for (auto& x : Bm) x = ud(rng);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 4);
    cudaMalloc(&dB, Bm.size() * 4);
    cudaMalloc(&dC, C0.size() * 4);
    cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bm.data(), Bm.size() * 4, cudaMemcpyHostToDevice);

    gemm_cpu(A.data(), Bm.data(), C0.data(), m, k, n);

    // 朴素版对答案
    {
        dim3 blk(16, 16), grd((n + 15) / 16, (m + 15) / 16);
        gemm_naive<<<grd, blk>>>(dA, dB, dC, m, k, n);
        cudaMemcpy(C1.data(), dC, C1.size() * 4, cudaMemcpyDeviceToHost);
        cout << "朴素版 与 CPU 最大误差 = " << max_diff(C0, C1) << endl;
    }
    // 瓷砖版对答案
    {
        dim3 blk(TS, TS), grd((n + TS - 1) / TS, (m + TS - 1) / TS);
        gemm_tiled<<<grd, blk>>>(dA, dB, dC, m, k, n);
        cudaMemcpy(C1.data(), dC, C1.size() * 4, cudaMemcpyDeviceToHost);
        cout << "瓷砖版 与 CPU 最大误差 = " << max_diff(C0, C1) << endl;
    }

    // 2) 大矩阵跑分：1024×1024×1024（20.9 亿次乘加）
    m = k = n = 1024;
    vector<float> B2(m * k), B3(k * n), C2(m * n);
    for (auto& x : B2) x = ud(rng);
    for (auto& x : B3) x = ud(rng);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaMalloc(&dA, B2.size() * 4);
    cudaMalloc(&dB, B3.size() * 4);
    cudaMalloc(&dC, C2.size() * 4);
    cudaMemcpy(dA, B2.data(), B2.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B3.data(), B3.size() * 4, cudaMemcpyHostToDevice);
    double flops = 2.0 * m * k * n;

    auto t0 = chrono::steady_clock::now();
    gemm_cpu(B2.data(), B3.data(), C2.data(), m, k, n);
    auto t1 = chrono::steady_clock::now();
    cout << "CPU   : " << chrono::duration<double>(t1 - t0).count() << " 秒  ("
         << flops / chrono::duration<double>(t1 - t0).count() / 1e9 << " GFLOPS)" << endl;

    dim3 blk(TS, TS), grd((n + TS - 1) / TS, (m + TS - 1) / TS);
    gemm_naive<<<grd, blk>>>(dA, dB, dC, m, k, n);           // 先热个身
    cudaDeviceSynchronize();
    t0 = chrono::steady_clock::now();
    gemm_naive<<<grd, blk>>>(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();
    t1 = chrono::steady_clock::now();
    cout << "GPU朴素: " << chrono::duration<double>(t1 - t0).count() << " 秒  ("
         << flops / chrono::duration<double>(t1 - t0).count() / 1e9 << " GFLOPS)" << endl;

    gemm_tiled<<<grd, blk>>>(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();
    t0 = chrono::steady_clock::now();
    gemm_tiled<<<grd, blk>>>(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();
    t1 = chrono::steady_clock::now();
    cout << "GPU瓷砖: " << chrono::duration<double>(t1 - t0).count() << " 秒  ("
         << flops / chrono::duration<double>(t1 - t0).count() / 1e9 << " GFLOPS)" << endl;

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
