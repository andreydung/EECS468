#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__device__ const int INPUT_WIDTH_PADDED = (INPUT_WIDTH + 128) & 0xFFFFFF80;

// Simple approach
__global__ void simple_histogram(uint32_t *bin, uint32_t *data, const int dataN) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	if (pos % INPUT_WIDTH_PADDED < INPUT_WIDTH) {
		uint32_t item = data[pos];
		atomicAdd(&(bin[item]), 1);
	}
}

__global__ void convertToChar(uint32_t *bin, uint8_t* out) {
	if (bin[threadIdx.x] > 255)
		out[threadIdx.x] = 255;
	else
		out[threadIdx.x] = bin[threadIdx.x];
}

void opt_2dhisto_simple(uint32_t *input, size_t height, size_t width, uint32_t* bins, uint8_t* result)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
	cudaMemset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(int));
	
	const int ARRAY_SIZE = INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80);	

	simple_histogram<<<ARRAY_SIZE/128, 128>>>(bins, input, height * width);
	cudaThreadSynchronize();

	convertToChar<<<1, HISTO_HEIGHT * HISTO_WIDTH>>>(bins, result);	
}

// Use shared memory
__global__ void histogram(uint32_t* result, uint32_t *data) {
	const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;
	
	__shared__ unsigned int s_Hist[BIN_COUNT];
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
		s_Hist[pos] = 0;
	__syncthreads();

	for(int pos = globalTid; pos < INPUT_HEIGHT * INPUT_WIDTH_PADDED; pos += numThreads) {
		if (pos % INPUT_WIDTH_PADDED < INPUT_WIDTH) {
			uint32_t item = data[pos];
			atomicAdd(s_Hist + item, 1);
		}	
	}
	__syncthreads();
	
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
		atomicAdd(result + pos, s_Hist[pos]);
}

void opt_2dhisto_fromslide(uint32_t* input, size_t height, size_t width, uint32_t* result32, uint8_t* result) {
	const int ARRAY_SIZE = INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80);	
	histogram<<<ARRAY_SIZE/1024, 1024 >>>(result32, input);
	convertToChar<<<1, HISTO_HEIGHT * HISTO_WIDTH>>>(result32, result);	
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

void FreeDevice(void* p) {
	cudaFree(p);	
}
