#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.


__global__ void prescan(float *outArray, float *inArray, int numElements) {
	__shared__ float temp[2 * BLOCK_SIZE];
	
	int tid = threadIdx.x;
	int offset = 1;

	temp[2*tid] = inArray[2*tid];
	temp[2*tid + 1] = inArray[2*tid + 1];

	for(int d = numElements >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2*tid + 1) - 1;
			int bi = offset * (2*tid + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
			
	if (tid == 0) {temp[numElements - 1] = 0;}

	for (int d = 1; d < numElements; d *= 2) {
		offset >>= 1;
		__syncthreads();

		if (tid < d) {
			int ai = offset * (2*tid + 1) - 1;
			int bi = offset * (2*tid + 2) - 1;
			
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	
	__syncthreads();
	
	outArray[2*tid] = temp[2*tid];
	outArray[2*tid + 1] = temp[2*tid + 1];
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
	prescan<<<1, BLOCK_SIZE>>>(outArray, inArray, numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
