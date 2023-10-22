#include "cuda_runtime.h"
#include "cudaheader.cuh"

texture<int, 2> texIn;

void cpu_imgFiltering(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk) 
{
	for (size_t i=1; i<imgSize-1; i++)
		for (size_t j = 1; j < imgSize - 1; j++)
		{
            outputImg[(i - 1) * (imgSize - 2) + (j - 1)] = 
                fk.top * inputImg[(i - 1) * imgSize + j] +
                fk.bottom * inputImg[(i + 1) * imgSize + j] +
                fk.left * inputImg[i * imgSize + (j - 1)] +
                fk.right * inputImg[i * imgSize + (j + 1)] +
                fk.center * inputImg[i * imgSize + j] +
                fk.top_left * inputImg[(i - 1) *imgSize + (j - 1)] +
                fk.top_right * inputImg[(i - 1) * imgSize + (j + 1)] +
                fk.bottom_left * inputImg[(i + 1) * imgSize + (j - 1)] +
                fk.bottom_right * inputImg[(i + 1) * imgSize + (j + 1)];
		}
}

__global__ void global_filterKernel(int* d_outputImg, int* d_inputImg, filterKernel fk)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    d_outputImg[offset] = 
        fk.top_left * d_inputImg[x + y * blockDim.x * gridDim.x] +
        fk.top * d_inputImg[(x + 1) + y * blockDim.x * gridDim.x] +
        fk.top_right * d_inputImg[(x + 2) + y * blockDim.x * gridDim.x] +
        fk.left * d_inputImg[x + (y + 1) * blockDim.x * gridDim.x] +
        fk.center * d_inputImg[(x + 1) + (y + 1) * blockDim.x * gridDim.x] +
        fk.right * d_inputImg[(x + 2) + (y + 1) * blockDim.x * gridDim.x] +
        fk.bottom_left * d_inputImg[x + (y + 2) * blockDim.x * gridDim.x] +
        fk.bottom * d_inputImg[(x + 1) + (y + 2) * blockDim.x * gridDim.x] +
        fk.bottom_right * d_inputImg[(x + 2) + (y + 2) * blockDim.x * gridDim.x];
}

__global__ void global_filterKernel(int* d_outputImg, filterKernel fk)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    d_outputImg[offset] =
        fk.top_left * tex2D(texIn, x, y) +
        fk.top * tex2D(texIn, x + 1, y) +
        fk.top_right * tex2D(texIn, x + 2, y) +
        fk.left * tex2D(texIn, x, y + 1) +
        fk.center * tex2D(texIn, x + 1, y + 1) +
        fk.right * tex2D(texIn, x + 2, y + 1) +
        fk.bottom_left * tex2D(texIn, x, y + 2) +
        fk.bottom * tex2D(texIn, x + 1, y + 2) +
        fk.bottom_right * tex2D(texIn, x + 2, y + 2);
}

void  global_imgFiltering(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk)
{
    int* d_inputImg;
    int* d_outputImg;

    cudaMalloc((void**)&d_inputImg, imgDim * imgDim * sizeof(int));
    cudaMalloc((void**)&d_outputImg, (imgDim - 2) * (imgDim - 2) * sizeof(int));

    cudaMemcpy(d_inputImg, inputImg, imgDim * imgDim * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 blocks(imgDim / 32, imgDim / 32);
    dim3 threads(32, 32);

    global_filterKernel << <blocks, threads >> > (d_outputImg, d_inputImg, fk);

    cudaMemcpy(outputImg, d_outputImg, (imgDim - 2) * (imgDim - 2) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImg);
    cudaFree(d_outputImg);
}

void  texture_imgFiltering(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk)
{
    int* d_inputImg;
    int* d_outputImg;

    
   
    cudaMalloc((void**)&d_inputImg, imgDim * imgDim * sizeof(int));
    cudaMalloc((void**)&d_outputImg, (imgDim - 2) * (imgDim - 2) * sizeof(int));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture2D(NULL, texIn, d_inputImg, desc, imgDim, imgDim, sizeof(int) * imgDim);
    
    cudaUnbindTexture(texIn);
    cudaFree(d_inputImg);
    cudaFree(d_outputImg);
}