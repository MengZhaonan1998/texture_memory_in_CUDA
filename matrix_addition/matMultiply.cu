#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>

texture<int> texB;
texture<int, 2> tex2dB;
__constant__ int constB[16384];

__global__ void globalMultiplyKernel(int* dev_A, int* dev_B, int* dev_C, size_t matSize)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    while (x < matSize && y < matSize)
    {
        dev_C[x * matSize + y] = 0;
        for (size_t i = 0; i < matSize; i++)
            dev_C[x * matSize + y] += dev_A[x * matSize + i] * dev_B[i * matSize + y];
        
        x += gridDim.x * blockDim.x;
        y += gridDim.y * blockDim.y;
    }
}

__global__ void constantMultiplyKernel(int* dev_A, int* dev_C, size_t matSize)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    while (x < matSize && y < matSize)
    {
        dev_C[x * matSize + y] = 0;
        for (size_t i = 0; i < matSize; i++)
            dev_C[x * matSize + y] += dev_A[x * matSize + i] * constB[i * matSize + y];

        x += gridDim.x * blockDim.x;
        y += gridDim.y * blockDim.y;
    }
}

__global__ void texture1DMultiplyKernel(int* dev_A, int* dev_C, size_t matSize)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    while (x < matSize && y < matSize)
    {
        dev_C[x * matSize + y] = 0;
        for (size_t i = 0; i < matSize; i++)
            dev_C[x * matSize + y] += dev_A[x * matSize + i] * tex1Dfetch(texB, i * matSize + y);

        x += gridDim.x * blockDim.x;
        y += gridDim.y * blockDim.y;
    }
}

__global__ void texture2DMultiplyKernel(int* dev_A, int* dev_C, size_t matSize)
{
    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    while (x < matSize && y < matSize)
    {
        dev_C[x * matSize + y] = 0;
        for (size_t i = 0; i < matSize; i++)
            dev_C[x * matSize + y] += dev_A[x * matSize + i] * tex2D(tex2dB, i, y);

        x += gridDim.x * blockDim.x;
        y += gridDim.y * blockDim.y;
    }
}

void globalMatMultiply(int* A, int* B, int* C, size_t matSize)
{
    int* dev_A, * dev_B, * dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_A, A, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy(dev_B, B, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice cudaMalloc failed!");
        goto Error;
    }

    dim3 threads(16, 16, 1);
    dim3 blocks((matSize + 15) / 16, (matSize + 15) / 16, 1);
    globalMultiplyKernel << <blocks, threads >> > (dev_A, dev_B, dev_C, matSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, matSize * matSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void constantMatMultiply(int* A, int* B, int* C, size_t matSize)
{
    int* dev_A, * dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaMemcpyToSymbol(constB, B, matSize * matSize * sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Constant HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy(dev_A, A, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }

    dim3 threads(16, 16, 1);
    dim3 blocks((matSize + 15) / 16, (matSize + 15) / 16, 1);
    constantMultiplyKernel << <blocks, threads >> > (dev_A, dev_C, matSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, matSize * matSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaFree(dev_A);
    cudaFree(dev_C);
}

void texture1DMatMultiply(int* A, int* B, int* C, size_t matSize)
{
    int* dev_A, * dev_B, * dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_A, A, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy(dev_B, B, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }

    cudaBindTexture(NULL, texB, dev_B, matSize * matSize * sizeof(int));

    dim3 threads(16, 16, 1);
    dim3 blocks((matSize + 15) / 16, (matSize + 15) / 16, 1);
    texture1DMultiplyKernel << <blocks, threads >> > (dev_A, dev_C, matSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, matSize * matSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaUnbindTexture(texB);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

void texture2DMatMultiply(int* A, int* B, int* C, size_t matSize)
{
    int* dev_A, * dev_B, * dev_C;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_A, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    // For texture2D, we have to make sure that array is aligned to memory?
    size_t pitch;
    cudaMallocPitch((void**)&dev_B, &pitch, matSize * sizeof(int), matSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch failed!");
        goto Error;
    }
//   cudaStatus = cudaMalloc((void**)&dev_B, matSize * matSize * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//       fprintf(stderr, "cudaMalloc failed!");
//       goto Error;
//    }
    cudaStatus = cudaMalloc((void**)&dev_C, matSize * matSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_A, A, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice Memcpy failed!");
        goto Error;
    }
    cudaMemcpy2D(dev_B, pitch, B, matSize * sizeof(int), matSize * sizeof(int), matSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice2D failed!");
        goto Error;
    }
//    cudaMemcpy(dev_B, B, matSize * matSize * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "HostToDevice Memcpy failed!");
//        goto Error;
//    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, tex2dB, dev_B, desc, matSize, matSize, pitch);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "texture binding failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    dim3 threads(16, 16, 1);
    dim3 blocks((matSize + 15) / 16, (matSize + 15) / 16, 1);
    texture2DMultiplyKernel << <blocks, threads >> > (dev_A, dev_C, matSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, matSize * matSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DeviceToHost failed!");
        goto Error;
    }

Error:
    cudaUnbindTexture(tex2dB);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}