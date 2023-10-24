#include <iostream>
#include <cstdlib>
#include <random>
#include <time.h>
#include "cudaheader.cuh"

#define DIM 1024

void matrix_coutput(double* matrix, size_t N, size_t xy_offset, size_t xy_length)
{
    std::cout << "The matrix is: \n";
    for (size_t i = xy_offset; i < xy_offset + xy_length; i++)
    {
        for (size_t j = xy_offset; j < xy_offset + xy_length; j++)
        {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}

// This program is designed to test the performance of matrix addition computed by different techniques 
void test_imgfilter()
{
    /*
    size_t N = 4096; // Problem size

    double* h_A = new double[N * N]{0.0};
    double* h_B = new double[N * N]{0.0};
    double* h_C = new double[N * N]{0.0};

    //double lower_bound = 0;
    //double upper_bound = 100;
    //std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    //std::default_random_engine re;
    for (size_t i = 0; i < N * N; i++)
    {   
        h_A[i] = 2.0 * i;//unif(re);
        h_B[i] = 3.0 * i;//unif(re);
    }

    clock_t tic;
    clock_t toc;

    tic = clock();
    std::cout << "-------- CPU starts to compute A+B=C -------- \n";
    cpu_addition(h_A, h_B, h_C, N);
    toc = clock();
    std::cout << "CPU computation completed! It took " << (double)(toc-tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    matrix_coutput(h_C, N, 5, 3);

    tic = clock();
    std::cout << "-------- GPU starts to compute A+B=C using the global memory --------\n";
    global_cudaAdd(h_A, h_B, h_C, N);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    matrix_coutput(h_C, N, 5, 3);
    */

    size_t N = 128; 
    filterKernel fk; 
    
    fk.top = 2;
    fk.bottom = 1;
    fk.left = 4;
    fk.right = 5;
    fk.center = 0;
    fk.top_left = 8;
    fk.top_right = 6;
    fk.bottom_left = 7;
    fk.bottom_right = 3;

    clock_t tic;
    clock_t toc;

    int* inputImg = new int[N * N]{ 0 };
    int* outputImg = new int[(N - 2) * (N - 2)]{ 0 };
    for (size_t i = 0; i < N * N; i++)inputImg[i] = i;

    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    for (size_t i = 0; i < 10; i++) global_imgFiltering(outputImg, inputImg, N, fk);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    for (size_t i = 0; i < 10; i++) texture_imgFiltering(outputImg, inputImg, N, fk);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    for (size_t i = 0; i < 10; i++) constant_imgFiltering(outputImg, inputImg, N, fk);
    toc = clock();
    std::cout << "Computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] inputImg;
    delete[] outputImg;
}

void test_matrixadd()
{
    size_t N = 8192; // Problem size

    int* h_A = new int[N]{ 0 };
    int* h_B = new int[N]{ 0 };
    int* h_C = new int[N]{ 0 };

    for (size_t i = 0; i < N; i++)
    {
        h_A[i] = 2 * i;
        h_B[i] = 3 * i;
        h_C[i] = 0;
    }

    clock_t tic;
    clock_t toc;

    tic = clock();
    std::cout << "-------- Global memory -------- \n";
    for (size_t i = 0; i < 10; i++) global_cudaAdd(h_A, h_B, h_C, N);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Constant memory -------- \n";
    for (size_t i = 0; i < 10; i++) constant_cudaAdd(h_A, h_B, h_C, N);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    tic = clock();
    std::cout << "-------- Texture memory -------- \n";
    for (size_t i = 0; i < 10; i++) texture_cudaAdd(h_A, h_B, h_C, N);
    toc = clock();
    std::cout << "GPU computation completed! It took " << (double)(toc - tic) / CLOCKS_PER_SEC << " seconds" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main()
{
    test_matrixadd();

    return 0;
}