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
#include "nfft_timer.h"

typedef thrust::complex<float> cuComplex;


// some macros

#define KPI  float(3.1415926535897932384626433832795028841971693993751)
#define K2PI float(6.2831853071795864769252867665590057683943387987502)
#define K4PI float(12.5663706143591729538505735331180115367886775975004)
#define KE   float(2.7182818284590452353602874713526624977572470937000)


// define a size for the block array. This allows for 4kb shared memory
// 1-D used SIZE_X*SIZE_Y*SIZE_Z
// 2-D uses SIZE_X*SIZE_Y
// 3-D uses SIZE_X*SIZE_Y*SIZE_Z

#define SIZE1_X 1024
#define SIZE2_X 8
#define SIZE2_Y 8
#define SIZE3_X 8
#define SIZE3_Y 8
#define SIZE3_Z 8

#define SIZE_1D (SIZE1_X)
#define SIZE_2D (SIZE2_X*SIZE2_Y)
#define SIZE_3D (SIZE3_X*SIZE3_Y*SIZE3_Z)

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
	x = x* t;
	y = y * t;

	res.real(x);
	res.imag(y);

	return res;

}



//////////////////////////////////////////////////////////////////////////////////////////////
///
///         --- 1D NFT Direct Transform kernal horizontal (It 1) ------------------
///
/// define the 1D NFT Direct Transform Kernel 
/// Addapted from the NFFT package trafo_direct
/// Implemented in the horizontal configuration (i.e. N < M )
__global__ void cudaNFTDirect1D_horizontal(float* x, cuComplex* f, cuComplex* fhat, int N, int M)
{

	// taking x,fhat -> f,k
	// --------------------------------------------------------------
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
	__shared__ cuComplex fhatloc[SIZE_1D];

	float xloc;
	cuComplex floc;
	cuComplex omega;
	
	floc = 0;
	if( globalTid < M )
	{
		xloc = x[globalTid];
	}
	__syncthreads();

	int maxK = (N-1)/SIZE_1D + 1;

	// we need to do the following N times
	for( int k_Out = 0; k_Out < maxK; k_Out++ )
	{
	
		int tid = SIZE_1D * k_Out + threadIdx.x;
		// copy the data over
		if( tid < N )
		{
			fhatloc[threadIdx.x] = fhat[tid];
		}

		__syncthreads();
	
		if( globalTid < M )
		{	

			int maxN = 0;
			if( (k_Out + 1)*SIZE_1D > N )
			{
				maxN = N - k_Out*SIZE_1D;
			}
			else
			{
				maxN = SIZE_1D;
			}

			for ( int k_L = 0; k_L < maxN; k_L++)
			{			
				omega.real(0);
				omega.imag( -K2PI * ((float)(k_L + k_Out*SIZE_1D - N/2)) * xloc );


				floc += ( fhatloc[k_L] * cexpf( omega ) );
			}

		}

		__syncthreads();
	}

	__syncthreads();

	if( globalTid < M )
	{
		f[globalTid] = floc;
	}
	

}

//////////////////////////////////////////////////////////////////////////////////////////////
///
///         --- 1D NFT Direct Transform kernal Vertical (It 1) ------------------
///
/// define the 1D NFT Direct Transform Kernel 
/// Addapted from the NFFT package trafo_direct
/// Implemented in the vertical configuration (i.e. N > M )
__global__ void cudaNFTDirect1D_vertical(float* x, cuComplex* f, cuComplex* fhat, int N, int M)
{

	// taking x,fhat -> f,k
	// --------------------------------------------------------------
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

	const int globalTid = blockIdx.x;
	const int numThreads = blockDim.x * gridDim.x;
	const int blockSize = blockDim.x;
	const int tid = threadIdx.x;

	// shared x and fhat values
	__shared__ cuComplex fhatloc[SIZE_1D];
	__shared__ cuComplex tempSum; // temporary sum location for parallel reduce

	float xloc;
	cuComplex floc;
	cuComplex omega;
	
	floc = 0;
	if( globalTid < M )
	{
		xloc = x[globalTid];
	}
	__syncthreads();

	int maxK = (N-1)/SIZE_1D + 1;

	// we need to do the following N times
	for( int k_Out = 0; k_Out < maxK; k_Out++ )
	{
	
		int ktid = SIZE_1D * k_Out + threadIdx.x;
		
		int k_L = threadIdx.x;

		// copy the data over
		if( ktid < N )
		{
			fhatloc[threadIdx.x] = fhat[ktid];
		
				
			// apply the FT
			omega.real(0);
			omega.imag( -K2PI * ((float)(k_L + k_Out*SIZE_1D - N/2)) * xloc );


			fhatloc[k_L] = ( fhatloc[k_L] * cexpf( omega ) );
		
		}
		else
		{
			fhatloc[k_L] = 0;
		}

		__syncthreads();
	
		// now we do the reduce
		if (blockSize >= 1024) 
		{ 
			if (tid < 512) { fhatloc[tid] += fhatloc[tid + 512]; } __syncthreads(); 
		}  
		if (blockSize >=  512) 
		{ 
			if (tid < 256) { fhatloc[tid] += fhatloc[tid + 256]; } __syncthreads(); 
		}  
		if (blockSize >=  256) 
		{ 
			if (tid < 128) { fhatloc[tid] += fhatloc[tid + 128]; } __syncthreads(); 
		}  
		if (blockSize >=  128) 
		{ 
			if (tid <  64) { fhatloc[tid] += fhatloc[tid +  64]; } __syncthreads(); 
		}  
		if (tid < 32) 
		{  
			if (blockSize >= 64) fhatloc[tid] += fhatloc[tid + 32];   
			if (blockSize >= 32) fhatloc[tid] += fhatloc[tid + 16];   
			if (blockSize >= 16) fhatloc[tid] += fhatloc[tid + 8];   
			if (blockSize >=  8) fhatloc[tid] += fhatloc[tid + 4];   
			if (blockSize >=  4) fhatloc[tid] += fhatloc[tid + 2];   
			if (blockSize >=  2) fhatloc[tid] += fhatloc[tid + 1];  
		}  

		// now copy the data over
		if (tid == 0) 
		{  
			if( k_Out == 0 ) 
				tempSum = fhatloc[0];  
			else
				tempSum = tempSum + fhatloc[0];
		} 

		__syncthreads();
	}

	__syncthreads();

	if( globalTid < M && (tid == 0) )
	{
		f[globalTid] = tempSum;
	}
	

}

//////////////////////////////////////////////////////////////////////////////////////////////
///
///         --- 1D NFT Direct Adjoint kernal horizontal (It 1) ------------------
///
/// define the 1D NFT Direct Adjoint Kernel 
/// Addapted from the NFFT package trafo_direct
/// Implemented in the horizontal configuration (i.e. N < M )
__global__ void cudaNFTDirect1DA(float* x, cuComplex* f, cuComplex* fhat, int N, int M)
{

	// taking f,k -> x,fhat
	// --------------------------------------------------------------
	// #pragma omp parallel for default(shared) private(k_L)
        // for (k_L = 0; k_L < ths->N_total; k_L++)
        // {
        //	INT j;
        //	for (j = 0; j < ths->M_total; j++)
        //	{
        //		R omega = K2PI * ((R)(k_L - (ths->N_total/2))) * ths->x[j];
        //		f_hat[k_L] += f[j] * BASE(II * omega);
        //	}
        // }
	const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;
	

	// shared x and fhat values
	__shared__ cuComplex floc[SIZE_1D];
	__shared__ float     xloc[SIZE_1D];
	
	cuComplex fhatloc;
	cuComplex omega;
	
	fhatloc = 0;
	int k_L = globalTid;
	__syncthreads();

	int maxj = (M-1)/SIZE_1D + 1;

	// we need to do the following N times
	for( int j_Out = 0; j_Out < maxj; j_Out++ )
	{
	
		int tid = SIZE_1D * j_Out + threadIdx.x;
		// copy the data over
		if( tid < M )
		{
			floc[threadIdx.x] = f[tid];
			xloc[threadIdx.x] = x[tid];
		}

		__syncthreads();
	
		if( globalTid < N )
		{	

			int maxM = 0;
			if( (j_Out + 1)*SIZE_1D > M )
			{
				maxM = M - j_Out*SIZE_1D;
			}
			else
			{
				maxM = SIZE_1D;
			}

			for ( int j = 0; j < maxM; j++)
			{			
				omega.real(0);
				omega.imag( K2PI * ((float)(k_L - N/2)) * xloc[j] );


				fhatloc += ( f[j] * cexpf( omega ) );
			}

		}

		__syncthreads();
	}

	__syncthreads();

	if( globalTid < N )
	{
		fhat[globalTid] = fhatloc;
	}
	

}


#if 0
//////////////////////////////////////////////////////////////////////////////////////////////
///
///         --- 2D NFT Direct Transform kernal (It 1) ------------------
///
/// define the 2D NFT Direct Transform Kernel 
/// Addapted from the NFFT package trafo_direct
__global__ void cudaNFTDirect2D(float* x, cuComplex* f, cuComplex* fhat, int N, int M, int Nx, int Ny)
{

	// taking x,fhat -> f
	// ---------------------------------------------------
        // #pragma omp parallel for default(shared) private(j)
    	// for (j = 0; j < ths->M_total; j++)
    	// {
      	//	R x[ths->d], omega, Omega[ths->d + 1];
      	//	INT t, t2, k_L, k[ths->d];
      	//	Omega[0] = K(0.0);
      	//	for (t = 0; t < ths->d; t++)
      	//	{
        //		k[t] = -ths->N[t]/2;
        //		x[t] = K2PI * ths->x[j * ths->d + t];
        //		Omega[t+1] = ((R)k[t]) * x[t] + Omega[t];
      	//	}
      	//	omega = Omega[ths->d];
	//	
      	//	for (k_L = 0; k_L < ths->N_total; k_L++)
      	//	{
        //		f[j] += f_hat[k_L] * BASE(-II * omega);
        //		{
        //  			for (t = ths->d - 1; (t >= 1) && (k[t] == ths->N[t]/2 - 1); t--)
        //    				k[t]-= ths->N[t]-1;
	//			
        //			k[t]++;
	//			
        //			for (t2 = t; t2 < ths->d; t2++)
        //				Omega[t2+1] = ((R)k[t2]) * x[t2] + Omega[t2];
	//			
        //			omega = Omega[ths->d];
        //		}
      	//	}
    	// }

	const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	const int numThreads = blockDim.x * gridDim.x;
	

	// shared x and fhat values
	__shared__ cuComplex fhatloc[SIZE_2D];

	const int d = 2;
	float xloc[d];
	int   kloc[d];
	cuComplex floc;
	cuComplex omega;
	float Omega[d+1];
	
	
	floc = 0;
	if( globalTid < M )
	{
		Omega[0] = 0;

		xloc[0] = K2PI*x[globalTid*d];
		xloc[1] = K2PI*x[globalTid*d + 1];
		//xloc[2] = K2PI*x[globalTid*d + 2];

		kloc[0] = -Nx/2;
		kloc[1] = -Ny/2;
		//kloc[2] = -Nz/2;
		
		Omega[1] = kloc[0]*xloc[0];
		Omega[2] = kloc[1]*xloc[1] + Omega[1];	
		//Omega[3] = kloc[2]*xloc[2] + Omega[2];	
		
		omega.imag( Omega[d] );
	}
	__syncthreads();

	int maxK = (N-1)/SIZE_1D + 1;

	// we need to do the following N times
	for( int k_Out = 0; k_Out < maxK; k_Out++ )
	{
	
		int tid = SIZE_1D * k_Out + threadIdx.x;
		// copy the data over
		if( tid < N )
		{
			fhatloc[threadIdx.x] = fhat[tid];
		}

		__syncthreads();
	
		if( globalTid < M )
		{	

			int maxN = 0;
			if( (k_Out + 1)*SIZE_1D > N )
			{
				maxN = N - k_Out*SIZE_1D;
			}
			else
			{
				maxN = SIZE_1D;
			}

			for ( int k_L = 0; k_L < maxN; k_L++)
			{			
				omega.real(0);
				omega.imag( -K2PI * ((float)(k_L + k_Out*SIZE_1D - N/2)) * xloc );


				floc += ( fhatloc[k_L] * cexpf( omega ) );
			}

		}

		__syncthreads();
	}

	__syncthreads();

	if( globalTid < M )
	{
		f[globalTid] = floc;
	}
	

}

#endif 

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

void Cuda_NFFT_trafo_direct_1d(nfftf_plan* plan, bool horizontal)
{
	// lets do some profiling
	TIMER_INIT;
	float time0, time1, time2;
	START_TIMER;
	// allocate x, f, fhat on the device
	float* x_loc = plan->x;
	float* x_dev = (float*)AllocateDevice( plan->M_total * sizeof(float) );

	cuComplex* fhat_loc = new cuComplex[plan->N_total];
	cuComplex* fhat_dev = (cuComplex*)AllocateDevice( plan->N_total * sizeof(cuComplex) );
	
	cuComplex* f_loc = new cuComplex[plan->M_total];
	cuComplex* f_dev = (cuComplex*)AllocateDevice( plan->M_total * sizeof(cuComplex) );


		
	// zero out the fhat
	//cudaMemset(f_dev, 0, (plan->M_total)*sizeof(cuComplex));

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
	CopyToDevice( x_dev, x_loc, (plan->M_total) * sizeof(float) );
	CopyToDevice( fhat_dev, fhat_loc, (plan->N_total) * sizeof(cuComplex) );


	printf("Size N %i, Size M %i \n", plan->N_total, plan->M_total);
	END_TIMER;
	time0 = TIMER_DIFF;

	START_TIMER;
	// now we call the kernel (recall: <<< num Blocks, threads per block >>> )
	int numBlocks;
	int numTiles; 
	cudaError_t err = cudaSuccess;
	if( horizontal )
	{	
		numBlocks = (plan->M_total - 1)/SIZE_1D+ 1;
		numTiles = ((plan->N_total)-1)/SIZE_1D + 1;
	}
	else
	{
		numBlocks = plan->M_total;
		numTiles = ((plan->N_total)-1)/SIZE_1D + 1; 
	}
	printf("\nNum Blocks %i, size %i, Num Tiles %i\n", numBlocks, SIZE_1D, numTiles);

	if( horizontal )
		cudaNFTDirect1D_horizontal <<< numBlocks, SIZE_1D >>> ( x_dev, f_dev, fhat_dev, plan->N_total, plan->M_total );
	else
		cudaNFTDirect1D_vertical <<< numBlocks, SIZE_1D >>> ( x_dev, f_dev, fhat_dev, plan->N_total, plan->M_total );

		cudaThreadSynchronize();
        END_TIMER;
        time1 = TIMER_DIFF;
	
	err = cudaPeekAtLastError();

	if( err != cudaSuccess ) 
		printf("%s \n",  cudaGetErrorString(err) );	

	START_TIMER;
	// now lets get the data back
	CopyFromDevice( f_dev, f_loc, plan->M_total * sizeof(cuComplex) );

	//memset( f_loc, 0, (plan->M_total)*sizeof(cuComplex) );
	// copy back to the plan
#if CUDA_SAFE_COMPLEX 
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

	END_TIMER;
        time2 = TIMER_DIFF;

	printf("times: %E (%E) %E s\n", time0, time1, time2 );

}

void Cuda_NFFT_adjoint_direct_1d(nfftf_plan* plan, bool horizontal)
{
	// lets do some profiling
	TIMER_INIT;
	float time0, time1, time2;
	START_TIMER;
	// allocate x, f, fhat on the device
	float* x_loc = plan->x;
	float* x_dev = (float*)AllocateDevice( plan->M_total * sizeof(float) );

	cuComplex* fhat_loc = new cuComplex[plan->N_total];
	cuComplex* fhat_dev = (cuComplex*)AllocateDevice( plan->N_total * sizeof(cuComplex) );
	
	cuComplex* f_loc = new cuComplex[plan->M_total];
	cuComplex* f_dev = (cuComplex*)AllocateDevice( plan->M_total * sizeof(cuComplex) );


		
	// zero out the fhat
	//cudaMemset(f_dev, 0, (plan->M_total)*sizeof(cuComplex));

	// copy from fftw_complex to cuComplex
#if CUDA_SAFE_COMPLEX 
	for( int ii = 0; ii < plan->M_total; ii++ )
	{
		f_loc[ii].real( plan->f[ii][0] );
		f_loc[ii].imag( plan->f[ii][1] );
	}
#else
	memcpy( f_loc, plan->f, 2*sizeof(float)*(plan->M_total) );
#endif	

	// now lets copy the f and x to the device
	CopyToDevice( x_dev, x_loc, (plan->M_total) * sizeof(float) );
	CopyToDevice( f_dev, f_loc, (plan->M_total) * sizeof(cuComplex) );


	printf("Size N %i, Size M %i \n", plan->N_total, plan->M_total);
	END_TIMER;
	time0 = TIMER_DIFF;

	START_TIMER;
	// now we call the kernel (recall: <<< num Blocks, threads per block >>> )
	int numBlocks;
	int numTiles; 
	cudaError_t err = cudaSuccess;
	
	numBlocks = (plan->N_total - 1)/SIZE_1D+ 1;
	numTiles = ((plan->M_total)-1)/SIZE_1D + 1;
	
	printf("\nNum Blocks %i, size %i, Num Tiles %i\n", numBlocks, SIZE_1D, numTiles);

	cudaNFTDirect1DA <<< numBlocks, SIZE_1D >>> ( x_dev, f_dev, fhat_dev, plan->N_total, plan->M_total );
	

	cudaThreadSynchronize();
        END_TIMER;
        time1 = TIMER_DIFF;

	err = cudaPeekAtLastError();

	if( err != cudaSuccess ) 
		printf("%s \n",  cudaGetErrorString(err) );	


	START_TIMER;
	// now lets get the data back
	CopyFromDevice( fhat_dev, fhat_loc, plan->N_total * sizeof(cuComplex) );

	//memset( f_loc, 0, (plan->M_total)*sizeof(cuComplex) );
	// copy back to the plan
#if CUDA_SAFE_COMPLEX 
        for( int ii = 0; ii < plan->N_total; ii++ )
        {
                plan->f_hat[ii][0] = fhat_loc[ii].real();
		plan->f_hat[ii][1] = fhat_loc[ii].imag();
	}
#else
	printf("here 2 \n");
        memcpy( fhat_loc, plan->fhat, 2*sizeof(float)*(plan->N_total) );
#endif
	// dealloc the memory on the device
	FreeDevice( x_dev );
	FreeDevice( f_dev );
	FreeDevice( fhat_dev );

	// delete local memory
	delete [] f_loc;
	delete [] fhat_loc;

	END_TIMER;
        time2 = TIMER_DIFF;

	printf("times: %E (%E) %E s\n", time0, time1, time2 );

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



