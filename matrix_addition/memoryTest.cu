#include "cuda_runtime.h"
#include "cudaheader.cuh"
#include <stdio.h>

texture<int> texIn;
//__constant__ int constIn[16384];

__global__ void global_filterKernel(int* dev_outputImg, int* dev_inputImg, size_t imgSize, filterKernel fk)
{
    int x = 1+ threadIdx.x + blockIdx.x * blockDim.x;
    int y = 1+ threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
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
    int x = 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int y = 1 + threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    int left = offset - 1;
    int right = offset + 1;
    int top = offset - imgSize;
    int bottom = offset + imgSize;
    
    if (x < (imgSize - 1) && y < (imgSize - 1))
    {
        dev_outputImg[offset] =
            fk.top * tex1Dfetch(texIn, top) +
            fk.left * tex1Dfetch(texIn, left) +
            fk.center * tex1Dfetch(texIn, offset) +
            fk.right * tex1Dfetch(texIn, right) +
            fk.bottom * tex1Dfetch(texIn, bottom);
    }
}

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
        fprintf(stderr, "HostToDevice cudaMemcpy failed!");
        goto Error;
    }

    //cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    //cudaBindTexture2D(0, texIn, dev_inputImg, desc, imgSize, imgSize, sizeof(int) * imgSize);
    cudaBindTexture(NULL, texIn, dev_inputImg, imgSize * imgSize * sizeof(int));

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
        fprintf(stderr, "DeviceToHost cudaMemcpy failed!");
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

