void cpu_addition(double* h_A, double* h_B, double* h_C, size_t N);
void global_cudaAdd(int* A, int* B, int* C, size_t N);
void constant_cudaAdd(int* A, int* B, int* C, size_t N);
void texture_cudaAdd(int* A, int* B, int* C, size_t N);

typedef struct {
	int top;
	int bottom;
	int left;
	int right;
	int center;
} filterKernel;

void  textureAccess(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk);
void  globalAccess(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk);

void  globalFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize);
void  textureFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize);