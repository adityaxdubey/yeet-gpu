#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int ind=threadIdx.x+blockIdx.x*blockDim.x;
    if (ind<N && input[ind]==K){
        atomicAdd(output,1);
    }
}

extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output, N, K);
    cudaDeviceSynchronize();
}