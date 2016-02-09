/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality 1-D accessing and combining memory
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel2(Matrix M, Matrix N, Matrix P)
{
	__shared__ float tileM[TILE_DIM*TILE_DIM*2];
	//__shared__ float tileN[TILE_DIM*TILE_DIM];

	unsigned int N_Tile = (M.width - 1)/TILE_DIM + 1; //ceil of integer division
	unsigned int tileOffset = TILE_DIM*TILE_DIM;

	unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	

	float Pval = 0;
	for (int k = 0; k < N_Tile; k++) 
	{
		// bring data from global memory into shared memory
		if ((yIndex < M.height) && (k*TILE_DIM + threadIdx.x) < M.width) 
		{
			tileM[threadIdx.y*TILE_DIM + threadIdx.x] = M.elements[yIndex*M.width + k*TILE_DIM + threadIdx.x];
		} 
		else 
		{
			tileM[threadIdx.y*TILE_DIM + threadIdx.x] = 0;
		}	
		
		if ((k*TILE_DIM + threadIdx.y) < N.height && (xIndex < N.width)) 
		{
			tileM[tileOffset + threadIdx.y*TILE_DIM + threadIdx.x] = N.elements[(k*TILE_DIM + threadIdx.y)*N.width + xIndex]; 
		} 
		else 
		{
			tileM[tileOffset + threadIdx.y*TILE_DIM + threadIdx.x] = 0;
		}

		__syncthreads();
	
		// tiled multiplication
		for (int i = 0; i < TILE_DIM; i++) 
		{
			Pval += tileM[threadIdx.y*TILE_DIM + i] * tileM[tileOffset + i*TILE_DIM + threadIdx.x];
		}
		__syncthreads();
	}

	if ((yIndex < P.height) && (xIndex < P.width)) 
	{
		P.elements[yIndex*P.width + xIndex] = Pval;
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
