#include <cstdio>
int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        printf("cudaGetDeviceCount 失败: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("GPU 数量: %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("[%d] %s | 显存 %.1f GB | 算力 %d.%d\n",
               i, p.name, p.totalGlobalMem / 1073741824.0,
               p.major, p.minor);
    }
    return 0;
}
