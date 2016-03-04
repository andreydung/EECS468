#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

// define the NFFTPlan
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define NFFT_PRECISION_SINGLE
#define __float128 float
#include "../include/nfft3mp.h"



/*
kernel/nfft/nfft.c:#define BASE(x) CEXP(x)
kernel/nfft/nfft.c:        f[j] += f_hat[k_L] * BASE(-II * omega);
kernel/nfft/nfft.c:        f[j] += f_hat[k_L] * BASE(-II * omega);
kernel/nfft/nfft.c:          f_hat[k_L] += f[j] * BASE(II * omega);
kernel/nfft/nfft.c:          f_hat[k_L] += f[j] * BASE(II * omega);
kernel/nfft/nfft.c:        f_hat[k_L] += f[j] * BASE(II * omega);
kernel/nfft/nfft.c:        f_hat[k_L] += f[j] * BASE(II * omega);
double _Complex cexp(double _Complex z);
*/

void CopyToDevice(void* device, void* host, size_t size) 
{
	cudaError_t err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) 
	{
		printf("Error copy to device: %s \n", cudaGetErrorString(err));
		exit(-1);
	}
}

void CopyFromDevice(void* device, void* host, size_t size) 
{
	cudaError_t err = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) 
	{
		printf("Error copy from device: %s \n", cudaGetErrorString(err));
		exit(-1);
	}
}

void* AllocateDevice(size_t size) 
{
	void* out;
	cudaError_t err = cudaMalloc(&out, size);	
	if (err != cudaSuccess) 
	{
		printf("Error allocating: %s \n", cudaGetErrorString(err));
		exit(-1);
	}
	return out;
}

void FreeDevice(void* p) 
{
	cudaFree(p);	
}


void Cuda_NFFT_trafo_1d(nfft_plan* plan)
{
}

void Cuda_NFFT_trafo_2d(nfft_plan* plan)
{
}



