#include <iostream>
#include <cstdlib>
#include <random>
#include <time.h>
#include "cudaAdd.cuh"

void cpu_addition(double* h_A, double* h_B, double* h_C, size_t N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_C[i * N + j] = h_A[i * N + j] + h_B[i * N + j];
};

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
int main()
{
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

    return 0;
}

