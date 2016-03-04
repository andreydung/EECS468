#ifndef NFFT_CUDA
#define NFFT_CUDA

void CopyToDevice(void* device, void* host, size_t size);
void CopyFromDevice(void* device, void* host, size_t size);
void* AllocateDevice(size_t size);
void FreeDevice(void* p);

#endif