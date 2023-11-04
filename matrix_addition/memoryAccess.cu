#include "cuda_runtime.h"
#include "cudaheader.cuh"
#include <stdio.h>

/*
| 0, 2, 4, ...
| 1, 3, 5, ...
*/

texture<int, 2> texIn;
__constant__ int constIn[16384];

__global__ void globalAccessKernel(int* d_Input, int colSize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int offset1 = colSize * (tid % 2) + (tid / 2);
    int offset2 = offset1 + colSize * (1 - 2 * (tid % 2)) + (tid / 2);
    int useless;
    if (tid < 2 * colSize)
	    useless = d_Input[offset1] * d_Input[offset1] + d_Input[offset2] * d_Input[offset2];
}

__global__ void textureAccessKernel(int colSize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int useless;
    if (tid < 2 * colSize)
        useless = tex2D(texIn, tid / 2, tid % 2) * tex2D(texIn, tid / 2, tid % 2) + tex2D(texIn, tid / 2, tid % 2 + 1 - 2 * (tid % 2)) * tex2D(texIn, tid / 2, tid % 2 + 1 - 2 * (tid % 2));
}

__global__ void constantAccessKernel(int colSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset1 = colSize * (tid % 2) + (tid / 2);
    int offset2 = offset1 + colSize * (1 - 2 * (tid % 2)) + (tid / 2);
    int useless;
    if (tid < 2 * colSize)
        useless = constIn[offset1] * constIn[offset1] + constIn[offset2] * constIn[offset2];
}

void globalAccess(int* h_Input, int colSize)
{
    int* d_Input;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_Input, colSize * 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(d_Input, h_Input, colSize * 2 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 blocks = (colSize * 2 + 31) / 32;
    dim3 threads = 32;

    globalAccessKernel << <blocks, threads >> > (d_Input, colSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(d_Input);
}

void textureAccess(int* h_Input, int colSize)
{
    int* d_Input;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_Input, 2 * colSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
   
    size_t pitch;
    cudaStatus = cudaMallocPitch((void**)&d_Input, &pitch, colSize * sizeof(int), 2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocPitch failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy2D(d_Input, pitch, h_Input, colSize * sizeof(int), colSize * sizeof(int), 2, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "HostToDevice2D failed!");
        goto Error;
    }
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaStatus = cudaBindTexture2D(NULL, texIn, d_Input, desc, colSize, 2, pitch);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "texture binding failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    dim3 blocks = (colSize * 2 + 31) / 32;
    dim3 threads = 32;

    textureAccessKernel << <blocks, threads >> > (colSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaUnbindTexture(texIn);
    cudaFree(d_Input);
}

void constantAccess(int* h_Input, int colSize)
{
    int* d_Input;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(constIn, h_Input, colSize * 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Constant HostToDevice Memcpy failed!");
        goto Error;
    }

    dim3 blocks = (colSize * 2 + 31) / 32;
    dim3 threads = 32;

    constantAccessKernel << <blocks, threads >> > (colSize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    return;
}