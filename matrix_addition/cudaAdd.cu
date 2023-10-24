#include "cuda_runtime.h"

texture<int> texA;
texture<int> texB;

__constant__ int constA[8192];
__constant__ int constB[8192];

__global__ void global_arrayAdd(int* d_A, int* d_B, int* d_C)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    d_C[tidx] = d_A[tidx] + d_B[tidx];
}

__global__ void constant_arrayAdd(int* d_C)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    d_C[tidx] = constA[tidx] + constB[tidx];
}

__global__ void texture_arrayAdd(int* d_C)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    d_C[tidx] = tex1Dfetch(texA, tidx) + tex1Dfetch(texB, tidx);
}

void global_cudaAdd(int* A, int* B, int* C, size_t N)
{
    // Initialize device pointers.
    int* d_A, *d_B, *d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(32, 1, 1);
    dim3 gridSize(N / 32, 1, 1);

    // Launch CUDA kernel.
    global_arrayAdd << <gridSize, blockSize >> > (d_A, d_B, d_C);

    // Copy result array c back to host memory
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void constant_cudaAdd(int* A, int* B, int* C, size_t N)
{
    // Initialize device pointers.
    int* d_C;

    // Allocate device memory.
    cudaMemcpyToSymbol("constA", A, 8192 * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol("constB", B, 8192 * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Calculate blocksize and gridsize.
    dim3 blockSize(32, 1, 1);
    dim3 gridSize(N / 32, 1, 1);

    // Launch CUDA kernel.
    constant_arrayAdd << <gridSize, blockSize >> > (d_C);

    // Copy result array c back to host memory
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_C);
}

void texture_cudaAdd(int* A, int* B, int* C, size_t N)
{
    // Initialize device pointers.
    int* d_C;

    // Allocate device memory. 
    cudaMalloc((void**)&d_C, N * sizeof(int));
    cudaBindTexture(NULL, texA, A, N);
    cudaBindTexture(NULL, texB, B, N);

    // Calculate blocksize and gridsize.
    dim3 blockSize(32, 1, 1);
    dim3 gridSize(N / 32, 1, 1);

    // Launch CUDA kernel.
    texture_arrayAdd << <gridSize, blockSize >> > (d_C);

    // Copy result array c back to host memory
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaUnbindTexture(texA);
    cudaUnbindTexture(texB);
    cudaFree(d_C);
}