#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024

// Lab4: Host Helper Functions (allocate your own data structure...)

// Lab4: Device Functions

// Lab4: Kernel Functions
__global__ void preScan(float *outArray, float *inArray, float *sums) {
// numElements must be blockDim.x * BLOCK_SIZE,
	
	__shared__ float temp[BLOCK_SIZE];
	
	int bid = BLOCK_SIZE * blockIdx.x;
	int tid = threadIdx.x;
	int offset = 1;
	
	temp[2*tid]     = inArray[bid + 2*tid];
	temp[2*tid + 1] = inArray[bid + 2*tid + 1];

	for(int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2*tid + 1) - 1;
			int bi = offset * (2*tid + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	__syncthreads();	
	if (tid == 0) {	
		sums[blockIdx.x] = temp[BLOCK_SIZE - 1];	
		temp[BLOCK_SIZE - 1] = 0;
	}

	for (int d = 1; d < BLOCK_SIZE; d *= 2) {
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
	
	outArray[bid + 2*tid]     = temp[2*tid];
	outArray[bid + 2*tid + 1] = temp[2*tid + 1];
}

__global__ void preScanSingle(float *outArray, float *inArray, int numElements) {
	__shared__ float temp[BLOCK_SIZE];
	
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

__global__ void addIncr(float* outArray, float* Incrs) {
	outArray[blockDim.x * blockIdx.x + threadIdx.x] += Incrs[blockIdx.x];	
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

void prescanArray(float *outArray, float *inArray, int numElements)
{
	int Nblocks = (numElements + BLOCK_SIZE - 1)/BLOCK_SIZE;
	
	float* d_sums;	
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_sums, sizeof(float) * Nblocks));
	
	printf("Size of Nblocks: %d \n", Nblocks);	
	
	preScan<<<Nblocks, BLOCK_SIZE/2>>>(outArray, inArray, d_sums);

	if (Nblocks <= BLOCK_SIZE/2) {
		float* d_incrs;
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_incrs, sizeof(float) * Nblocks ));
		preScanSingle<<<1, BLOCK_SIZE/2>>>(d_incrs, d_sums, Nblocks);
		addIncr<<<2*Nblocks, BLOCK_SIZE >>>(outArray, d_incrs);
	}
	else {
		
	}
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
