#include "cuda_runtime.h"
#include "cudaheader.cuh"
#include <stdio.h>

texture<int, 2> texMask;
__constant__ int constMask[4096];

__global__ void globalConvKernel(int* dev_inputImg, int* dev_convMask, size_t imgSize, size_t maskSize)
{
    int x = maskSize / 2 + threadIdx.x + blockIdx.x * blockDim.x;
    int y = maskSize / 2 + threadIdx.y + blockIdx.y * blockDim.y;

    while (x < imgSize - maskSize / 2 && y < imgSize - maskSize / 2)
    {
        for(size_t i = 0; i < maskSize; i++)
            for (size_t j = 0; j < maskSize; j++)
            {
                dev_inputImg[x + y * blockDim.x * gridDim.x] += 
                    dev_inputImg[(x + i - maskSize / 2) + (y + j - maskSize / 2) * blockDim.x * gridDim.x] * dev_convMask[i * maskSize + j];
            }
        x += blockDim.x * gridDim.x;
        y += blockDim.y * gridDim.y;
    }
}

__global__ void textureConvKernel(int* dev_inputImg, size_t imgSize, size_t maskSize)
{
    int x = maskSize / 2 + threadIdx.x + blockIdx.x * blockDim.x;
    int y = maskSize / 2 + threadIdx.y + blockIdx.y * blockDim.y;

    while (x < imgSize - maskSize / 2 && y < imgSize - maskSize / 2)
    {
        for (size_t i = 0; i < maskSize; i++)
            for (size_t j = 0; j < maskSize; j++)
            {
                dev_inputImg[x + y * blockDim.x * gridDim.x] +=
                    dev_inputImg[(x + i - maskSize / 2) + (y + j - maskSize / 2) * blockDim.x * gridDim.x] * tex2D(texMask, i, j);
            }
        x += blockDim.x * gridDim.x;
        y += blockDim.y * gridDim.y;
    }
}

__global__ void constantConvKernel(int* dev_inputImg, size_t imgSize, size_t maskSize)
{
    int x = maskSize / 2 + threadIdx.x + blockIdx.x * blockDim.x;
    int y = maskSize / 2 + threadIdx.y + blockIdx.y * blockDim.y;

    while (x < imgSize - maskSize / 2 && y < imgSize - maskSize / 2)
    {
        for (size_t i = 0; i < maskSize; i++)
            for (size_t j = 0; j < maskSize; j++)
            {
                dev_inputImg[x + y * blockDim.x * gridDim.x] +=
                    dev_inputImg[(x + i - maskSize / 2) + (y + j - maskSize / 2) * blockDim.x * gridDim.x] * constMask[i * maskSize + j];
            }
        x += blockDim.x * gridDim.x;
        y += blockDim.y * gridDim.y;
    }
}

void  globalFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize)
{
    int* dev_inputImg;
    int* dev_convMask;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_convMask, maskSize * maskSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputImg, inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_convMask, convMask, maskSize * maskSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 blocks((imgSize - maskSize + 31) / 32, (imgSize - maskSize + 31) / 32);
    dim3 threads(32, 32);

    globalConvKernel << <blocks, threads >> > (dev_inputImg, dev_convMask, imgSize, maskSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(inputImg, dev_inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_inputImg);
}

void  textureFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize)
{
    int* dev_inputImg;
    int* dev_convMask;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // For texture2D, we have to make sure that array is aligned to memory?
    size_t pitch;
    cudaStatus = cudaMallocPitch((void**)&dev_convMask, &pitch, maskSize * sizeof(int), maskSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inputImg, inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy2D(dev_convMask, pitch, convMask, maskSize * sizeof(int), maskSize * sizeof(int), maskSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice2D failed!");
        goto Error;
    }
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, texMask, dev_convMask, desc, maskSize, maskSize, pitch);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "texture binding failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    dim3 blocks((imgSize - maskSize + 31) / 32, (imgSize - maskSize + 31) / 32);
    dim3 threads(32, 32);

    textureConvKernel << <blocks, threads >> > (dev_inputImg, imgSize, maskSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(inputImg, dev_inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaUnbindTexture(texMask);

Error:
    cudaFree(dev_inputImg);
}

void constantFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize)
{
    int* dev_inputImg;
    int* dev_convMask;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputImg, imgSize * imgSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_inputImg, inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(constMask, convMask, maskSize * maskSize * sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Constant HostToDevice Memcpy failed!");
        goto Error;
    }
    
    dim3 blocks((imgSize - maskSize + 31) / 32, (imgSize - maskSize + 31) / 32);
    dim3 threads(32, 32);

    constantConvKernel << <blocks, threads >> > (dev_inputImg, imgSize, maskSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(inputImg, dev_inputImg, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_inputImg);
}