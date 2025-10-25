#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=N-i-1;
    if (i<j){
        float tmp=input[i];
        input[i]=input[j];
        input[j]=tmp; }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

    reverse_array<<<blocksPerGrid,threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}