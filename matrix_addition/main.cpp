#include <iostream>
#include <cstdlib>
#include <random>
#include <time.h>
#include "cudaheader.cuh"

void ConvolutionFiltering()
{
    size_t imgSize   = 2048;   // image size
    size_t maskSize = 128*4;     // mask size

    // Allocate host memory
    int* inputImg  = new int[imgSize * imgSize]{ 0 };
    int* convMask = new int[maskSize * maskSize]{ 0 };
    
    // Initialization
    for (size_t i = 0; i < imgSize * imgSize; i++)inputImg[i] = rand();
    for (size_t i = 0; i < maskSize * maskSize; i++)convMask[i] = rand();

    // Timer
    clock_t tic;
    clock_t toc;

    // Test::Global memory
    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    //for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    globalFiltering(inputImg, convMask, imgSize, maskSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // Test::Texture memory
    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    //for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    textureFiltering(inputImg, convMask, imgSize, maskSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;
    
    delete[] inputImg;
    delete[] convMask;
}

void MemoryAccess()
{
    size_t imgSize = 4096;   
    filterKernel fk;
    fk.left = 1;
    fk.right = 2;
    fk.top = 3;
    fk.bottom = 4;
    fk.center = -1;

    int* inputImg = new int[imgSize * imgSize]{ 0 };
    int* outputImg = new int[imgSize * imgSize]{ 0 };

    for (size_t i = 0; i < imgSize * imgSize; i++)inputImg[i] = 1;

    clock_t tic;
    clock_t toc;

    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    globalAccess(outputImg, inputImg, imgSize, fk);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    textureAccess(outputImg, inputImg, imgSize, fk);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] inputImg;
    delete[] outputImg;
}

void arrayAdd()
{
    size_t arraySize = 8192; 

    int* h_A = new int[arraySize]{ 0 };
    int* h_B = new int[arraySize]{ 0 };
    int* h_C = new int[arraySize]{ 0 };

    for (size_t i = 0; i < arraySize; i++)
    {
        h_A[i] = 2 * i;
        h_B[i] = 3 * i;
        h_C[i] = 0;
    }

    clock_t tic;
    clock_t toc;

    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    for (size_t i = 0; i < 100; i++) globalArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    for (size_t i = 0; i < 100; i++) constantArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    for (size_t i = 0; i < 100; i++) textureArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main()
{
    GetDeviceInfo();
    
    //MemoryAccess();
    //ConvolutionFiltering();
    arrayAdd();

    return 0;
}