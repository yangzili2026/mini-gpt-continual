# 把 lesson4_gpt.cpp 改造成 lesson5_gpt.cu：
# 1) 万能头 -> 标准头 + cuda_runtime.h
# 2) 三个 CPU 矩阵乘改名为 *_cpu，新增 GPU kernel + 同名分发函数
# 3) 第4个命令行参数 cpu/gpu 切换（默认 gpu），方便对照实验
import io

src = io.open('lesson4_gpt.cpp', encoding='utf-8').read()

# ---- 1) 头文件 ----
old_inc = '#include <bits/stdc++.h>'
new_inc = '''#include <cstdio>
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
#include <cuda_runtime.h>'''
assert old_inc in src
src = src.replace(old_inc, new_inc, 1)

# ---- 2) CPU 三个函数改名 ----
src = src.replace('void gemm(const float* A, const float* Bm, float* C, int m, int k, int n) {',
                  'void gemm_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {', 1)
src = src.replace('void gemm_bt(const float* A, const float* Bm, float* C, int m, int k, int n) {',
                  'void gemm_bt_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {', 1)
src = src.replace('void gemm_at(const float* A, const float* Bm, float* C, int m, int k, int n) {',
                  'void gemm_at_cpu(const float* A, const float* Bm, float* C, int m, int k, int n) {', 1)

# ---- 3) 在 gemm_at_cpu 结束后插入 GPU 区块 ----
anchor = '''            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
        }
}
void bias_add('''
assert anchor in src
gpu_block = '''            for (int j = 0; j < n; j++) Crow[j] += a * Brow[j];
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
const size_t GMAX = 512 * 512;              // 我们最大的矩阵也就 512×512
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

void bias_add('''
src = src.replace(anchor, gpu_block, 1)

# ---- 4) main 里加开关解析 + 提示行 ----
old_main = '    cout << "参数量: " << g.nparams() / 1000 << " K" << endl;'
assert old_main in src
new_main = '''    cout << "参数量: " << g.nparams() / 1000 << " K" << endl;
    if (argc > 4 && string(argv[4]) == "cpu") USE_GPU = false;   // 第4参数 cpu = 纯CPU对照
    cout << "矩阵乘: " << (USE_GPU ? "GPU (RTX 4070)" : "CPU 对照") << endl;
    if (USE_GPU) { float* p = nullptr; cudaMalloc(&p, 16); cudaFree(p); }  // 预热显卡'''    
src = src.replace(old_main, new_main, 1)

# ---- 5) 标题注释改一下 ----
src = src.replace('C++ 造大语言模型 · 第4课：完整迷你 GPT',
                  'C++ 造大语言模型 · 第5课：迷你 GPT + GPU 矩阵乘', 1)

io.open('lesson5_gpt.cu', 'w', encoding='utf-8-sig').write(src)
print('lesson5_gpt.cu written,', len(src), 'chars')
