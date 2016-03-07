#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>

// define the NFFTPlan
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_SAFE_COMPLEX 1


#define NFFT_PRECISION_SINGLE
#define __float128 float
#include "../include/nfft3mp.h"


typedef thrust::complex<float> cuComplex;


// some macros

#define KPI  float(3.1415926535897932384626433832795028841971693993751)
#define K2PI float(6.2831853071795864769252867665590057683943387987502)
#define K4PI float(12.5663706143591729538505735331180115367886775975004)
#define KE   float(2.7182818284590452353602874713526624977572470937000)

//////////////////////////////////////////////////////////////////////////////////////////////
///
///      ---- Complex Exponential Function -----------
///
/// define the cexp function
/// taken from: https://devtalk.nvidia.com/default/topic/505308/complex-number-exponential-function/
/// performs exp(z) = exp(x + iy) = exp(x) * exp (iy) = exp(x) * (cos(y) + i sin(y))
__device__ __forceinline__ cuComplex cexpf (cuComplex z)
{

	cuComplex res;
	float t = expf (z.real());
	float x = 0, y = 0;

	sincosf (z.imag(), &y, &x);
	x *= t;
	y *= t;

	res.real(x);
	res.imag(y);

	return res;

}

//////////////////////////////////////////////////////////////////////////////////////////////
///
///         --- 1D NFT Direct Transform kernal (It 1) ------------------
///
/// define the 1D NFT Direct Transform Kernel 
/// Addapted from the NFFT package trafo_direct
__global__ void cudaNFTDirect1D(float* x, cuComplex* f, cuComplex* fhat, int N, int M)
{

	// taking x,fhat -> f
	// --------------------------------------------------------------
	// memset(f, 0, (size_t)(ths->M_total) * sizeof(C));
	// #pragma omp parallel for default(shared) private(j)
	// for (j = 0; j < ths->M_total; j++)
	// {
	// 	INT k_L;
	//	for (k_L = 0; k_L < ths->N_total; k_L++)
	//	{
	//		R omega = K2PI * ((R)(k_L - ths->N_total/2)) * ths->x[j];
	//		f[j] += f_hat[k_L] * BASE(-II * omega);
	//	}
	// }

	const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;


	// shared x and fhat values
	__shared__ cuComplex fhatloc[32];

	float xloc = x[globalTid];
	cuComplex floc;
	cuComplex omega;
	int k_L;
	
	floc.real(0);
	floc.imag(0);
	
	
	
	// copy the data over
	if( globalTid < N )
	{
		fhatloc[globalTid] = fhat[globalTid];
	}

	__syncthreads();
	
	if( globalTid < M )
	{

	
		for ( k_L = 0; k_L < N; k_L++)
		{			
			omega.real(0);
			omega.imag( -K2PI * ((float)(k_L - N/2)) * xloc );
			floc = floc + fhat[k_L] * cexpf( omega );
		}


		
	}

	__syncthreads();


	if( globalTid < M )
	{
		f[globalTid] = floc;
	}
	//if( globalTid < M && globalTid < N)
	//{
	//	f[globalTid] = fhatloc[globalTid];
	//}

}

// Define the cuda NFT Plan


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

void Cuda_NFFT_trafo_direct_1d(nfftf_plan* plan)
{
	// allocate x, f, fhat on the device
	float* x_loc = new float[plan->N_total];
	float* x_dev = (float*)AllocateDevice( plan->N_total * sizeof(float) );

	cuComplex* fhat_loc = new cuComplex[plan->N_total];
	cuComplex* fhat_dev = (cuComplex*)AllocateDevice( plan->N_total * sizeof(cuComplex) );
	
	cuComplex* f_loc = new cuComplex[plan->M_total];
	cuComplex* f_dev = (cuComplex*)AllocateDevice( plan->M_total * sizeof(cuComplex) );


		
	// zero out the fhat
	cudaMemset(f_dev, 0, (plan->M_total)*sizeof(cuComplex));

// copy from fftw_complex to cuComplex
#if CUDA_SAFE_COMPLEX 
	for( int ii = 0; ii < plan->N_total; ii++ )
	{
		fhat_loc[ii].real( plan->f_hat[ii][0] );
		fhat_loc[ii].imag( plan->f_hat[ii][1] );
	}
#else
	memcpy( fhat_loc, plan->f_hat, 2*sizeof(float)*(plan->N_total) );
#endif	

	// now lets copy the f and x to the device
	CopyToDevice( x_dev, x_loc, plan->N_total * sizeof(float) );
	CopyToDevice( fhat_dev, fhat_loc, plan->N_total * sizeof(cuComplex) );

	// now we call the kernel
	cudaNFTDirect1D <<< plan->M_total, 1 >>> ( x_dev, f_dev, fhat_dev, plan->N_total, plan->M_total );

	cudaThreadSynchronize();

	// now lets get the data back
	CopyFromDevice( f_dev, f_loc, plan->M_total * sizeof(cuComplex) );

	//memset( f_loc, 0, (plan->M_total)*sizeof(cuComplex) );
	// copy back to the plan
#if CUDA_SAFE_COMPLEX 
	printf("here \n");
        for( int ii = 0; ii < plan->M_total; ii++ )
        {
                plan->f[ii][0] = f_loc[ii].real();
                plan->f[ii][1] = f_loc[ii].imag();
        }
#else
	printf("here 2 \n");
        memcpy( f_loc, plan->f, 2*sizeof(float)*(plan->M_total) );
#endif
	// dealloc the memory on the device
	FreeDevice( x_dev );
	FreeDevice( f_dev );
	FreeDevice( fhat_dev );

	// delete local memory
	delete [] f_loc;
	delete [] fhat_loc;
	delete [] x_loc;
}

void Cuda_NFFT_trafo_direct_2d(nfftf_plan* plan)
{
}



void Cuda_NFFT_trafo_1d(nfftf_plan* plan)
{
	cuComplex cc;
}

void Cuda_NFFT_trafo_2d(nfftf_plan* plan)
{
}



