#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>

void GetDeviceInfo()
{
	CUresult result;
	result = cuInit(0);
	CUdevice device;
	result = cuDeviceGet(&device, 0);

	int texture1Dwidth;
	int texture2Dwidth;
	int texture2Dheight;

	result = cuDeviceGetAttribute(&texture1Dwidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device);
	printf("CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = %d KBs\n", texture1Dwidth/1024);

	result = cuDeviceGetAttribute(&texture2Dwidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device);
	printf("CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = %d KBs\n", texture2Dwidth/1024);
	
	result = cuDeviceGetAttribute(&texture2Dheight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device);
	printf("CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = %d KBs\n", texture2Dheight/1024);
}