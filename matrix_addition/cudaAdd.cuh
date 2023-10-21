void kernel(double* A, double* B, double* C, int arraySize);
void global_cudaAdd(double* A, double* B, double* C, size_t N);
void constant_cudaAdd(double* A, double* B, double* C, size_t N);
void texture_cudaAdd(double* A, double* B, double* C, size_t N);