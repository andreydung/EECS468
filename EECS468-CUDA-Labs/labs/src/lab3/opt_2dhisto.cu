#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__device__ const int INPUT_WIDTH_PADDED = (INPUT_WIDTH + 128) & 0xFFFFFF80;

__global__ void simple_histogram(int *bin, uint32_t *data, const int dataN) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	if (pos % INPUT_WIDTH_PADDED < INPUT_WIDTH) {
		uint32_t item = data[pos] % BIN_COUNT;
		atomicAdd(&(bin[item]), 1);
	}
}

__global__ void clampToChar(int *bin, uint8_t* out) {
	if (bin[threadIdx.x] > 255)
		out[threadIdx.x] = 255;
	if (bin[threadIdx.x] < 0)
		out[threadIdx.x] = 0;
	else
		out[threadIdx.x] = bin[threadIdx.x];
}

/*
__global__ void histogram(uint *d_Result, uint *d_Data, int dataN) {
	const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;
	
	__shared__ uint s_Hist[BIN_COUNT];
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
		s_Hist[pos] = 0;
	__syncthreads();

	for(int post = globalTid; pos < dataN; pos += numThreads) {
		uint data4 = d_Data[pos];
		atomicAdd (s_Hist + (data4 >> 0)  & 0xFFU, 1);
		atomicAdd (s_Hist + (data4 >> 8)  & 0xFFU, 1);
		atomicAdd (s_Hist + (data4 >> 16) & 0xFFU, 1);
		atomicAdd (s_Hist + (data4 >> 24) & 0xFFU, 1);
	}
	__syncthreads();
	
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
		atomicAdd(d_Result + pos, s_Hist[pos]);
}
*/

void opt_2dhisto_simple(uint32_t *input, size_t height, size_t width, int* bins, uint8_t* result)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
	cudaMemset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(int));
	
	const int ARRAY_SIZE = INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80);	

	simple_histogram<<<ARRAY_SIZE/128, 128>>>(bins, input, height * width);
	cudaThreadSynchronize();

	clampToChar<<<1, HISTO_HEIGHT * HISTO_WIDTH>>>(bins, result);	
}

/* Include below the implementation of any other functions you need */
void CopyToDevice(void* device, void* host, size_t size) {
	cudaError_t err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("Error copy to device: %s \n", cudaGetErrorString(err));
		exit(-1);
	}
}

void CopyFromDevice(void* device, void* host, size_t size) {
	cudaError_t err = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("Error copy from device: %s \n", cudaGetErrorString(err));
		exit(-1);
	};
}

void* AllocateDevice(size_t size) {
	void* out;
	cudaError_t err = cudaMalloc(&out, size);	
	if (err != cudaSuccess) {
		printf("Error allocating: %s \n", cudaGetErrorString(err));
		exit(-1);
	};
	return out;
}
