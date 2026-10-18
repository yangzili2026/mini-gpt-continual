extern "C" __global__ void hello(float* out) {
    out[threadIdx.x] = threadIdx.x * 2.0f;
}
