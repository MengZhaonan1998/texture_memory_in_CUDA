#include <iostream>
#include <time.h>
#include "cuda_runtime.h"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}

__global__ void global_cudaMatAdd(double* A, double* B, double* C, size_t N) 
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    while (tidx < N && tidy < N)
    {
        C[tidx * N + tidy] = A[tidx * N + tidy] + B[tidx * N + tidy];
        tidx += gridDim.x * blockDim.x;
        tidy += gridDim.y * blockDim.y;
    }
}

void kernel(double* A, double* B, double* C, int arraySize) {

    // Initialize device pointers.
    double* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, arraySize * sizeof(double));
    cudaMalloc((void**)&d_B, arraySize * sizeof(double));
    cudaMalloc((void**)&d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}

void global_cudaAdd(double* A, double* B, double* C, size_t N)
{
    // Initialize device pointers.
    double* d_A, *d_B, *d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_B, N * N * sizeof(double));
    cudaMalloc((void**)&d_C, N * N * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(32, 32, 1);
    dim3 gridSize(1024, 1024, 1);

    // Launch CUDA kernel.
    global_cudaMatAdd << <gridSize, blockSize >> > (d_A, d_B, d_C, N);

    // Copy result array c back to host memory
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}