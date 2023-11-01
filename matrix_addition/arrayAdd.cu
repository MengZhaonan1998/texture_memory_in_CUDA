#include "cuda_runtime.h"
#include <stdio.h>

texture<int> texA;
texture<int> texB;

__constant__ int constA[8192];
__constant__ int constB[8192];

__global__ void globalAddKernel(int* dev_A, int* dev_B, int* dev_C, int arraySize)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    while (tidx < arraySize)
    {
        dev_C[tidx] = dev_A[tidx] + dev_B[tidx];
        tidx += blockDim.x * gridDim.x;
    }
}

__global__ void constantAddKernel(int* dev_C, int arraySize)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    while (tidx < arraySize)
    {
        dev_C[tidx] = constA[tidx] + constB[tidx];
        tidx += blockDim.x * gridDim.x;
    }
}

__global__ void textureAddKernel(int* dev_C, int arraySize)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    while (tidx < arraySize)
    {
        dev_C[tidx] = tex1Dfetch(texA, tidx) + tex1Dfetch(texB, tidx);
        tidx += blockDim.x * gridDim.x;
    }   
}

void globalArrayAdd(int* A, int* B, int* C, size_t arraySize)
{
    int* dev_A, *dev_B, *dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_A, A, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy(dev_B, B, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice cudaMalloc failed!");
        goto Error;
    }

    dim3 threads(128, 1, 1);
    dim3 blocks((arraySize + 127) / 128, 1, 1);
    globalAddKernel << <blocks, threads >> > (dev_A, dev_B, dev_C, arraySize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

    Error:
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void textureArrayAdd(int* A, int* B, int* C, size_t arraySize)
{
    int* dev_A, * dev_B, * dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_A, A, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy(dev_B, B, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice cudaMalloc failed!");
        goto Error;
    }

    cudaBindTexture(NULL, texA, dev_A, arraySize * sizeof(int));
    cudaBindTexture(NULL, texB, dev_B, arraySize * sizeof(int));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Texture binding failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    dim3 threads(128, 1, 1);
    dim3 blocks((arraySize + 127) / 128, 1, 1);
    textureAddKernel << <blocks, threads >> > (dev_C, arraySize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaUnbindTexture(texA);
    cudaUnbindTexture(texB);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void constantArrayAdd(int* A, int* B, int* C, size_t arraySize)
{
    int* dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    /*cudaStatus = cudaMalloc((void**)&dev_A, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }*/
    cudaStatus = cudaMalloc((void**)&dev_C, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpyToSymbol(constA, A, 8192 * sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Constant HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpyToSymbol(constB, B, 8192 * sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Constant HostToDevice cudaMalloc failed!");
        goto Error;
    }

    dim3 threads(128, 1, 1);
    dim3 blocks((arraySize + 127) / 128, 1, 1);
    constantAddKernel << <blocks, threads >> > (dev_C, arraySize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaFree(dev_C);
}
