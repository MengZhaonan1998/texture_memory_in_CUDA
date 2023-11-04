#include <iostream>
#include <cstdlib>
#include <random>
#include <time.h>
#include "cudaheader.cuh"

void printMat(int* matrix, size_t matSize)
{
    for (size_t i = 0; i < matSize; i++)
    {
        for (size_t j = 0; j < matSize; j++)
        {
            std::cout << matrix[i * matSize + j] << "  ";
        }
        std::cout << "\n";
    }
}

void ConvolutionFiltering()
{
    size_t imgSize = 1024;   // image size
    size_t maskSize = 32;     // mask size

    // Allocate host memory
    int* inputImg  = new int[imgSize * imgSize]{ 0 };
    int* convMask = new int[maskSize * maskSize]{ 0 };
    
    // Timer
    clock_t tic;
    clock_t toc;

    // Initialization
    for (size_t i = 0; i < imgSize * imgSize; i++)inputImg[i] = i;
    for (size_t i = 0; i < maskSize * maskSize; i++)convMask[i] = i;

    // Test::Global memory
    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    //for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    globalFiltering(inputImg, convMask, imgSize, maskSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printMat(inputImg, imgSize);

    for (size_t i = 0; i < imgSize * imgSize; i++)inputImg[i] = i;
    for (size_t i = 0; i < maskSize * maskSize; i++)convMask[i] = i;

    // Test::Texture memory
    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    //for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    textureFiltering(inputImg, convMask, imgSize, maskSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printMat(inputImg, imgSize);

    for (size_t i = 0; i < imgSize * imgSize; i++)inputImg[i] = i;
    for (size_t i = 0; i < maskSize * maskSize; i++)convMask[i] = i;

    // Test::Constant memory
    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    //for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    constantFiltering(inputImg, convMask, imgSize, maskSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printMat(inputImg, imgSize);

    delete[] inputImg;
    delete[] convMask;
}

void MemoryAccess()
{
    size_t colSize = 2048;   

    int* h_Input = new int[2 * colSize]{ 0 };
    
    for (size_t i = 0; i < 2 * colSize; i++)h_Input[i] = i;

    clock_t tic;
    clock_t toc;
  
    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    for (size_t i = 0; i < 1000; i++) globalAccess(h_Input, colSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    for (size_t i = 0; i < 1000; i++) textureAccess(h_Input, colSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    for (size_t i = 0; i < 1000; i++) constantAccess(h_Input, colSize);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] h_Input;
}

void arrayAdd()
{
    size_t arraySize = 8192; 

    int* h_A = new int[arraySize]{ 0 };
    int* h_B = new int[arraySize]{ 0 };
    int* h_C = new int[arraySize]{ 0 };

    clock_t tic;
    clock_t toc;

    for (size_t i = 0; i < arraySize; i++)
    {
        h_A[i] = 2 * i;
        h_B[i] = 3 * i;
        h_C[i] = 0;
    }

    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    for (int i = 0; i < 1000; i++) globalArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    for (size_t i = 0; i < arraySize; i++)
    {
        h_A[i] = 2 * i;
        h_B[i] = 3 * i;
        h_C[i] = 0;
    }

    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    for (int i = 0; i < 1000; i++) constantArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    for (size_t i = 0; i < arraySize; i++)
    {
        h_A[i] = 2 * i;
        h_B[i] = 3 * i;
        h_C[i] = 0;
    }

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    for (int i = 0; i < 1000; i++) textureArrayAdd(h_A, h_B, h_C, arraySize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

void matrixMultiply()
{
    size_t matSize = 128;

    int* h_A = new int[matSize * matSize] { 0 };
    int* h_B = new int[matSize * matSize] { 0 };
    int* h_C = new int[matSize * matSize] { 0 };

    for (size_t i = 0; i < matSize; i++)
        for(size_t j = 0; j < matSize; j++)
        {
            h_A[i * matSize + j] = i + j;
            h_B[i * matSize + j] = i  - j;
            h_C[i * matSize + j] = 0;
        }

    clock_t tic;
    clock_t toc;

    tic = clock();
    std::cout << "-------- Matrix multiply :: Global memory -------- \n";
    for (int i = 0; i < 1000; i++)globalMatMultiply(h_A, h_B, h_C, matSize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;
    
    //printMat(h_C, matSize);

    tic = clock();
    std::cout << "-------- Matrix multiply :: Constant memory -------- \n";
    for (int i = 0; i < 1000; i++)constantMatMultiply(h_A, h_B, h_C, matSize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;
    
    //printMat(h_C, matSize);

    tic = clock();
    std::cout << "-------- Matrix multiply :: Texture1D memory -------- \n";
    for (int i = 0; i < 1000; i++)texture1DMatMultiply(h_A, h_B, h_C, matSize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printMat(h_C, matSize);

    tic = clock();
    std::cout << "-------- Matrix multiply :: Texture2D memory -------- \n";
    for (int i = 0; i < 1000; i++)texture2DMatMultiply(h_A, h_B, h_C, matSize);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;


    //printMat(h_C, matSize);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main()
{
    GetDeviceInfo();
    
    //ConvolutionFiltering();
    //arrayAdd();
    //MemoryAccess();
    matrixMultiply();

    return 0;
}