AM_CPPFLAGS = -I$(top_srcdir)/include
NVCC = nvcc

NVCC_INCLUDES = -I/usr/local/cuda/include -I/usr/local/cuda-7.0/include

# compiling flags
NVCC_CXXFLAGS = -O2 -G -dc

# the linking flags
NVCC_LINKFLAGS =  -L/usr/local/cuda/lib64 -lcudart

bin_PROGRAMS = test_cuda
test_cuda_SOURCES = test_cuda.cpp nfft_cuda.cu
test_cuda_LDFLAGS = gpu_link.o $(NVCC_LINKFLAGS) 
test_cuda_LDADD = $(top_builddir)/libnfft3@PREC_SUFFIX@.la @fftw3_LDFLAGS@ @fftw3_LIBS@
test_cuda_AR = $(NVCC) -lib -o

.cu.o :
	$(NVCC) $(NVCC_CXXFLAGS) $(NVCC_INCLUDES) -c -o $@ $<
	$(NVCC) -dlink -o gpu_link.o $@
