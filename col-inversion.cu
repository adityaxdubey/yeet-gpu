#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<width*height){
        image[4*i]=255-image[4*i];
        image[4*i+1]=255-image[4*i+1];
        image[4*i+2]=255-image[4*i+2];
    }
}
extern "C" void solve(unsigned char* image, int width,int height) {
    int threadsPerBlock=256;
    int blocksPerGrid=(width*height+threadsPerBlock-1)/threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}