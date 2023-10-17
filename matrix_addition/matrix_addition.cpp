#include <iostream>
#include <cstdlib>
#include <time.h>

void cpu_addition(double* h_A, double* h_B, double* h_C, int N)
{
    clock_t time;
    time = clock();
    std::cout << "-------- CPU starts to compute A+B=C --------" << "\n";
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_C[i * N + j] = h_A[i * N + j] + h_B[i * N + j];
    time = clock()-time;
    std::cout << "CPU computation completed! It took " << ((float)time) / CLOCKS_PER_SEC << " seconds" << std::endl;
};


// This program is designed to test the performance of matrix addition computed by different techniques 
int main()
{
    int N = 4096;
    double* h_A = new double[N * N];
    double* h_B = new double[N * N];
    double* h_C = new double[N * N];

    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = (double)(i+j);
            h_B[i * N + j] = (double)(i-j);
            h_C[i * N + j] = (double)0.0;
        }
    /*
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_A[i * N + j] << " ";
        }
        std::cout << "\n" << std::endl;
    }
    */

    cpu_addition(h_A, h_B, h_C, N);


    delete [] h_A;
    delete [] h_B;
    delete [] h_C;

    return 0;
}

