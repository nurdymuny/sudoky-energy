#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s\nCompute: sm_%d%d\n", p.name, p.major, p.minor);
    return 0;
}
