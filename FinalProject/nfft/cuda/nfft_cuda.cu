#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

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