AM_CPPFLAGS = -I$(top_srcdir)/include


noinst_PROGRAMS = test_cuda
test_cuda_SOURCES = test_cuda.c
test_cuda_LDADD = $(top_builddir)/libnfft3@PREC_SUFFIX@.la @fftw3_LDFLAGS@ @fftw3_LIBS@
