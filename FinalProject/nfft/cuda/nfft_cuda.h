#ifndef NFFT_CUDA
#define NFFT_CUDA

#include "nfft3mp.h"

void CopyToDevice(void* device, void* host, size_t size);
void CopyFromDevice(void* device, void* host, size_t size);
void* AllocateDevice(size_t size);
void FreeDevice(void* p);

void Cuda_NFFT_trafo_1d(nfft_plan* plan);
void Cuda_NFFT_trafo_2d(nfft_plan* plan);

#endif
