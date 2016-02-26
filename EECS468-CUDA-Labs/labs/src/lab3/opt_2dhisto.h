#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto_simple_1(uint32_t *input, size_t height, size_t width, uint32_t* bin, uint8_t* result);
void opt_2dhisto_shared_2(uint32_t* input, size_t height, size_t width, uint32_t* result32, uint8_t* result);
void CopyToDevice(void* device, void* host, size_t size);
void CopyFromDevice(void* device, void* host, size_t size);
void* AllocateDevice(size_t size);
void FreeDevice(void* p);

#endif
