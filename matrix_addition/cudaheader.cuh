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
	int top_left;
	int top_right;
	int bottom_left;
	int bottom_right;
} filterKernel;

void cpu_imgFiltering(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk);
void global_imgFiltering(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk);
void  texture_imgFiltering(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk);
void constant_imgFiltering(int* outputImg, int* inputImg, size_t imgDim, filterKernel fk);