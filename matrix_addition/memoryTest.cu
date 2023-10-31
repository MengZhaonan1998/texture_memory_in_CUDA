#include "cuda_runtime.h"
#include "cudaheader.cuh"
#include <stdio.h>

texture<int, 2> texIn;
//__constant__ int constIn[16384];

__global__ void global_filterKernel(int* dev_outputImg, int* dev_inputImg, size_t imgSize, filterKernel fk)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = (x + 1) + (y + 1) * blockDim.x * gridDim.x;
    if (x < (imgSize - 1) && y < (imgSize - 1))
    {
        dev_outputImg[offset] =
            fk.top * dev_inputImg[x + (y - 1) * blockDim.x * gridDim.x] +
            fk.left * dev_inputImg[(x - 1) + y * blockDim.x * gridDim.x] +
            fk.center * dev_inputImg[x + y * blockDim.x * gridDim.x] +
            fk.right * dev_inputImg[(x + 1) + y * blockDim.x * gridDim.x] +
            fk.bottom * dev_inputImg[x + (y + 1) * blockDim.x * gridDim.x];
    }      
}

__global__ void texture_filterKernel(int* dev_outputImg, size_t imgSize, filterKernel fk)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = (x + 1) + (y + 1) * blockDim.x * gridDim.x;
    if (x < (imgSize - 1) && y < (imgSize - 1))
    {
        dev_outputImg[offset] =
            fk.top * tex2D(texIn, x, y - 1) +
            fk.left * tex2D(texIn, x - 1, y) +
            fk.center * tex2D(texIn, x, y) +
            fk.right * tex2D(texIn, x + 1, y) +
            fk.bottom * tex2D(texIn, x, y + 1);
    }
}
/*
__global__ void const_filterKernel(int* d_outputImg, filterKernel fk)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    d_outputImg[offset] =
        fk.top_left * constIn[x + y * blockDim.x * gridDim.x] +
        fk.top * constIn[(x + 1) + y * blockDim.x * gridDim.x] +
        fk.top_right * constIn[(x + 2) + y * blockDim.x * gridDim.x] +
        fk.left * constIn[x + (y + 1) * blockDim.x * gridDim.x] +
        fk.center * constIn[(x + 1) + (y + 1) * blockDim.x * gridDim.x] +
        fk.right * constIn[(x + 2) + (y + 1) * blockDim.x * gridDim.x] +
        fk.bottom_left * constIn[x + (y + 2) * blockDim.x * gridDim.x] +
        fk.bottom * constIn[(x + 1) + (y + 2) * blockDim.x * gridDim.x] +
        fk.bottom_right * constIn[(x + 2) + (y + 2) * blockDim.x * gridDim.x];
}
*/


void  textureAccess(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk)
{
    int* dev_inputImg;
    int* dev_outputImg;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaMalloc((void**)&dev_inputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMalloc((void**)&dev_outputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputImg, inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, texIn, dev_inputImg, desc, imgSize, imgSize, sizeof(int) * imgSize);

    dim3 blocks((imgSize+31) / 32, (imgSize + 31) / 32);
    dim3 threads(32, 32);
    texture_filterKernel << <blocks, threads >> > (dev_outputImg, imgSize, fk);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(outputImg, dev_outputImg, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
    cudaUnbindTexture(texIn);
    cudaFree(dev_inputImg);
    cudaFree(dev_outputImg);
}

void  globalAccess(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk)
{
    int* dev_inputImg;
    int* dev_outputImg;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaMalloc((void**)&dev_inputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaMalloc((void**)&dev_outputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputImg, inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice cudaMemcpy failed!");
        goto Error;
    }

    dim3 blocks((imgSize + 31) / 32, (imgSize + 31) / 32);
    dim3 threads(32, 32);
    global_filterKernel << <blocks, threads >> > (dev_outputImg, dev_inputImg, imgSize, fk);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(outputImg, dev_outputImg, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_inputImg);
    cudaFree(dev_outputImg);
}

/*
void constantAccess(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk)
{
    int* d_outputImg;
    cudaMemcpyToSymbol("constIn", inputImg, 16384 * sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_outputImg, (imgDim - 2) * (imgDim - 2) * sizeof(int));

    dim3 blocks(imgDim / 32, imgDim / 32);
    dim3 threads(32, 32);

    const_filterKernel << <blocks, threads >> > (d_outputImg, fk);

    cudaMemcpy(outputImg, d_outputImg, (imgDim - 2) * (imgDim - 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_outputImg);
}
*/