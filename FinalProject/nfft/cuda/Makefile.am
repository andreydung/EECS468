AM_CPPFLAGS = -I$(top_srcdir)/include
NVCC = nvcc

# root path of cuda toolkits 
CUDA_ROOT_DIR = /usr/local/cuda

# the preprocessing flags passed to the compiler
NVCC_CPPFLAGS = -I$(CUDA_ROOT_DIR)/include \
                -I$(CUDA_ROOT_DIR)/samples/common/inc

# compiling flags
NVCC_CXXFLAGS = -O2 -g -G -dc \
                -gencode arch=compute_35,code=sm_35

# the linking flags
NVCC_LINKFLAGS = -gencode arch=compute_35,code=sm_35 \
                 -L/usr/local/cuda-5.0/lib64 -lcudart

noinst_PROGRAMS = test_cuda
test_cuda_SOURCES = test_cuda.c nfft_cuda.cu
test_cuda_LDFLAGS = $(NVCC_LINKFLAGS)
test_cuda_LDADD = $(top_builddir)/libnfft3@PREC_SUFFIX@.la @fftw3_LDFLAGS@ @fftw3_LIBS@

# the rule that can recognize *.cu files
# and compile the *.cu files to *.o files
.cu.o : ; $(NVCC) $(NVCC_CPPFLAGS) $(NVCC_CXXFLAGS) -c -o $@ $<
