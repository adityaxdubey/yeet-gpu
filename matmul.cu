#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    int c=blockIdx.y*blockDim.y+threadIdx.y;
    if (c<K && r<M){
        float sum=0.0;
        for(int i=0;i<N;i++){
            sum+=A[r*N+i]*B[i*K+c];
        }
        C[r*K+c]=sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
