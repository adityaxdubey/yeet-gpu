#include <cuda_runtime.h>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i<cols && j<rows){
        output[i*rows+j]=input[j*cols+i];
    }
}


extern "C" void solve(const float* input,float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols+threadsPerBlock.x-1)/threadsPerBlock.x,
                       (rows+threadsPerBlock.y-1)/threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid,threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}