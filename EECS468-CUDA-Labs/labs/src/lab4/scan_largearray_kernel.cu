#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 512 

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
	// numElemets does not need to be power of 2
	
	if (numElements > BLOCK_SIZE) return;
	
	__shared__ float temp[BLOCK_SIZE];
	
	int tid = threadIdx.x;
	int offset = 1;

	temp[2*tid] = (2*tid < numElements) ? inArray[2*tid]:0;
	temp[2*tid + 1] = (2*tid + 1 < numElements) ? inArray[2*tid + 1]:0;

	for(int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2*tid + 1) - 1;
			int bi = offset * (2*tid + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
			
	if (tid == 0) {temp[BLOCK_SIZE - 1] = 0;}

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
	
	if (2*tid < numElements) 
		outArray[2*tid] = temp[2*tid];
	if (2*tid + 1 < numElements)
		outArray[2*tid + 1] = temp[2*tid + 1];
	__syncthreads();
}

__global__ void addIncr(float* outArray, float* Incrs) {
	outArray[blockDim.x * blockIdx.x + threadIdx.x] += Incrs[blockIdx.x];	
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

void prescanArrayPowerTwo(float *outArray, float *inArray, int numElements)
{
	//check if power of two
	if (numElements & (numElements - 1) != 0) return;
	
	int Nblocks = (numElements + BLOCK_SIZE - 1)/BLOCK_SIZE;
	if (Nblocks <= BLOCK_SIZE>>1) {
		float* sums;	
		CUDA_SAFE_CALL(cudaMalloc((void**) &sums, sizeof(float) * Nblocks));
		float* incrs;
		CUDA_SAFE_CALL(cudaMalloc((void**) &incrs, sizeof(float) * Nblocks ));
	
		preScan<<<Nblocks, BLOCK_SIZE/2>>>(outArray, inArray, sums);
		preScanSingle<<<1, BLOCK_SIZE/2>>>(incrs, sums, Nblocks);
		addIncr<<<Nblocks, BLOCK_SIZE >>>(outArray, incrs);
	}
	else {
		float* middle;
		CUDA_SAFE_CALL(cudaMalloc((void**) &middle, sizeof(float) * Nblocks ));
		float* incrMiddle;
        CUDA_SAFE_CALL(cudaMalloc((void**) &incrMiddle, sizeof(float) * Nblocks ));

		int Nblocks2 = (Nblocks + BLOCK_SIZE - 1)/BLOCK_SIZE;	
		float* sums;
		CUDA_SAFE_CALL(cudaMalloc((void**) &sums, sizeof(float) * Nblocks2));
        float* incrs;
		CUDA_SAFE_CALL(cudaMalloc((void**) &incrs, sizeof(float) * Nblocks2));	
	
		preScan<<<Nblocks ,  BLOCK_SIZE/2>>>(outArray, inArray, middle);
		cudaThreadSynchronize();
		preScan<<<Nblocks2,  BLOCK_SIZE/2>>>(incrMiddle, middle, sums);
		cudaThreadSynchronize();
		preScanSingle<<<1,   BLOCK_SIZE/2>>>(incrs, sums, Nblocks2);
		cudaThreadSynchronize();
		addIncr<<<Nblocks2,BLOCK_SIZE>>>(incrMiddle, incrs);
		addIncr<<<Nblocks, BLOCK_SIZE>>>(outArray, incrMiddle);
	}
}

int floorPowerTwo(int n) {
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	return n - (n >> 1);
}

__global__ void addConstant(float* arr, float* c1, float* c2, int n) {
	int tid = threadIdx.x;
	if (tid < n) arr[tid] += *c1 + *c2;
}

void prescanArray(float *outArray, float *inArray, int numElements) {
	int n = floorPowerTwo(numElements);
	
	printf("Original and rounding size: %d and %d\n", numElements, n);	
	prescanArrayPowerTwo(outArray, inArray, n);

	if (n < numElements) {
		printf("Extra for padding\n");
		preScanSingle<<<1, BLOCK_SIZE/2>>>(outArray + n, inArray + n, numElements - n);	
		cudaThreadSynchronize();
	//	addIncr<<<1, numElements - n>>>(outArray + n, constant);
		addConstant<<<1, numElements - n>>>(outArray + n, outArray+ n - 1, inArray + n, numElements - n);
	}
}

// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
