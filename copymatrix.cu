#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int y=blockDim.x*blockIdx.x+threadIdx.x;
    if (y<N*N){
        B[y]=A[y];
    }
}

//a and b are device pointers
extern "C" void solve(const float* A, float* B, int N) {
    int total=N*N;
    int threadsPerBlock=256;
    int blocksPerGrid=(total+threadsPerBlock-1)/threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid,threadsPerBlock>>>(A,B,N);
    cudaDeviceSynchronize();
} 