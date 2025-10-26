#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output,int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N){
        output[i]=max(input[i],(float)0.0);}
}


extern "C" void solve(const float* input, float* output,int N) {
    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

    relu_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output,N);
    cudaDeviceSynchronize();
}
