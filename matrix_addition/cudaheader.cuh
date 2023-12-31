﻿typedef struct {
	int top;
	int bottom;
	int left;
	int right;
	int center;
} filterKernel;

void  textureAccessNO(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk);
void  globalAccessNO(int* outputImg, int* inputImg, size_t imgSize, filterKernel fk);

void  globalFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize);
void  textureFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize);
void constantFiltering(int* inputImg, int* convMask, size_t imgSize, size_t maskSize);

void globalArrayAdd(int* A, int* B, int* C, size_t arraySize);
void constantArrayAdd(int* A, int* B, int* C, size_t arraySize);
void textureArrayAdd(int* A, int* B, int* C, size_t arraySize);

void globalMatMultiply(int* A, int* B, int* C, size_t matSize);
void constantMatMultiply(int* A, int* B, int* C, size_t matSize);
void texture1DMatMultiply(int* A, int* B, int* C, size_t matSize);
void texture2DMatMultiply(int* A, int* B, int* C, size_t matSize);

void globalAccess(int* h_Input, int colSize);
void textureAccess(int* h_Input, int colSize);
void constantAccess(int* h_Input, int colSize);

void GetDeviceInfo();