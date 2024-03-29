/*
 * Copyright (c) 2002, 2015 Jens Keiner, Stefan Kunis, Daniel Potts
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* $Id: simple_test.c.in 4298 2015-01-15 10:24:37Z tovo $ */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <complex>


#define NFFT_PRECISION_SINGLE

#define USING_ADJOINT 1

#define INPUT_SET 4096

/*
#define TIMER_INIT timespec start_time, end_time
#define START_TIMER clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time)
#define END_TIMER clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time)
#define TIMER_DIFF elapsed_s( start_time, end_time )

float elapsed_s( timespec start, timespec end )
{
	float elapsed = end.tv_nsec - start.tv_nsec;
	elapsed = elapsed * 1.0E-9;
	return elapsed;
} 
*/
#include "nfft_timer.h"
#include "nfft3mp.h"
#include "nfft_cuda.h"

double compute_error(fftwf_complex* x, fftwf_complex* y, int len)
{
	double error = 0;

	for( int ii = 0; ii < len; ii++ )
	{
		double tempR = x[ii][0] - y[ii][0];
		double tempI = x[ii][1] - y[ii][1];
		error = error + tempR*tempR + tempI*tempI;
	}

	return error;
}


void simple_test_nfft_1d(bool sparse)
{
  nfftf_plan p;


  int N, M, K;

  if( sparse )
  {
	N = INPUT_SET;
	M = 16;
	K = 16;
  }
  else
  {
	N = INPUT_SET;
	M = 2*N;
	K = 16;

  }

  //timespec start_time, end_time;
  TIMER_INIT;

  const char *error_str;

  /** init an one dimensional plan */
  NFFT(init_1d)(&p, N, M);


  // declare the comparison vector
  fftwf_complex* vcomp = new fftwf_complex[p.M_total];

  /** init pseudo random nodes */
  NFFT(vrand_shifted_unit_double)(p.x, p.M_total);
 
  /** precompute psi, the entries of the matrix B */
  if (p.flags & PRE_ONE_PSI)
      NFFT(precompute_one_psi)(&p);

  /** init pseudo random Fourier coefficients and show them */
  NFFT(vrand_unit_complex)(p.f_hat,p.N_total);
  NFFT(vpr_complex)(p.f_hat, K, "given Fourier coefficients, vector f_hat");

  /** check for valid parameters before calling any trafo/adjoint method */
  error_str = NFFT(check)(&p);
  if (error_str != 0)
  {
    printf("Error in nfft module: %s\n", error_str);
    return;
  }

  /** direct trafo and show the result */
  START_TIMER;
  NFFT(trafo_direct)(&p);
  END_TIMER;
  NFFT(vpr_complex)(p.f, K, "ndft, vector f");
  printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), 0.0f  );

  // copy the complex data over
  for( int ii = 0; ii < p.M_total; ii++ )
  {
	vcomp[ii][0] = p.f[ii][0];
	vcomp[ii][1] = p.f[ii][1];
  }
  /** cuda direct trafo and show the result */
  START_TIMER;
  Cuda_NFFT_trafo_direct_1d(&p, true);
  END_TIMER;
  NFFT(vpr_complex)(p.f, K ,"cu_ndft[h], vector f");
  printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), compute_error(vcomp, p.f, p.M_total) );

  /** cuda direct trafo and show the result */
  if( sparse )
  {
  	START_TIMER;
  	Cuda_NFFT_trafo_direct_1d(&p, false);
  	END_TIMER;
  	NFFT(vpr_complex)(p.f, K ,"cu_ndft[v], vector f");
  	printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), compute_error(vcomp, p.f, p.M_total) );
  }

  /** approx. trafo and show the result */
  START_TIMER;
  NFFT(trafo)(&p);
  END_TIMER; 
  NFFT(vpr_complex)(p.f, K ,"nfft, vector f");
  printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), compute_error(vcomp, p.f, p.M_total) );

#if USING_ADJOINT

  delete [] vcomp;
  // reallocate for the adjoint
  vcomp = new fftwf_complex[p.N_total]; 

  /** approx. adjoint and show the result */
  START_TIMER;
  NFFT(adjoint_direct)(&p);
  END_TIMER; 
  NFFT(vpr_complex)(p.f_hat, K,"adjoint ndft, vector f_hat");
  printf(" took %E seconds.L2 %E \n", elapsed_s( start_time, end_time), 0.0 );
  
  // copy the complex data over
  for( int ii = 0; ii < p.N_total; ii++ )
  {
	vcomp[ii][0] = p.f_hat[ii][0];
	vcomp[ii][1] = p.f_hat[ii][1];
  }
  /** cuda direct trafo and show the result */
  START_TIMER;
  Cuda_NFFT_adjoint_direct_1d(&p, true);
  END_TIMER;
  NFFT(vpr_complex)(p.f_hat, K ," adjoint cu_ndft[h], vector f");
  printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), compute_error(vcomp, p.f_hat, p.N_total) );
 

  /** approx. adjoint and show the result */
  START_TIMER;
  NFFT(adjoint)(&p);
  END_TIMER; 
  NFFT(vpr_complex)(p.f_hat, K,"adjoint nfft, vector f_hat");
  printf(" took %E seconds. L2 %E \n", elapsed_s( start_time, end_time), compute_error(vcomp, p.f_hat, p.N_total));

#endif


  // free the comparison vector
  delete [] vcomp;

  /** finalise the one dimensional plan */
  NFFT(finalize)(&p);
}

static void simple_test_nfft_2d(void)
{
  int K, N[2], n[2], M;
  NFFT_R t0, t1;

  NFFT(plan) p;

  const char *error_str;

  N[0] = 32; n[0] = 64;
  N[1] = 14; n[1] = 32;
  M = N[0] * N[1];
  K = 16;

  t0 = NFFT(clock_gettime_seconds)();
  /** init a two dimensional plan */
  NFFT(init_guru)(&p, 2, N, M, n, 7,
     PRE_PHI_HUT| PRE_FULL_PSI| MALLOC_F_HAT| MALLOC_X| MALLOC_F |
     FFTW_INIT| FFT_OUT_OF_PLACE,
     FFTW_ESTIMATE| FFTW_DESTROY_INPUT);

  /** init pseudo random nodes */
  NFFT(vrand_shifted_unit_double)(p.x, p.d * p.M_total);

  /** precompute psi, the entries of the matrix B */
  if(p.flags & PRE_ONE_PSI)
    NFFT(precompute_one_psi)(&p);

  /** init pseudo random Fourier coefficients and show them */
  NFFT(vrand_unit_complex)(p.f_hat, p.N_total);

  t1 = NFFT(clock_gettime_seconds)();
  NFFT(vpr_complex)(p.f_hat,K, "given Fourier coefficients, vector f_hat (first few entries)");
  printf(" ... initialisation took %.2" NFFT__FES__ " seconds.\n",t1-t0);

  /** check for valid parameters before calling any trafo/adjoint method */
  error_str = NFFT(check)(&p);
  if (error_str != 0)
  {
    printf("Error in nfft module: %s\n", error_str);
    return;
  }

  /** direct trafo and show the result */
  t0 = NFFT(clock_gettime_seconds)();
  NFFT(trafo_direct)(&p);
  t1 = NFFT(clock_gettime_seconds)();
  NFFT(vpr_complex)(p.f, K, "ndft, vector f (first few entries)");
  printf(" took %.2" NFFT__FES__ " seconds.\n",t1-t0);

  /** approx. trafo and show the result */
  t0 = NFFT(clock_gettime_seconds)();
  NFFT(trafo)(&p);
  t1 = NFFT(clock_gettime_seconds)();
  NFFT(vpr_complex)(p.f, K, "nfft, vector f (first few entries)");
  printf(" took %.2" NFFT__FES__ " seconds.\n",t1-t0);


#if USING_ADJOINT

  /** direct adjoint and show the result */
  t0 = NFFT(clock_gettime_seconds)();
  NFFT(adjoint_direct)(&p);
  t1 = NFFT(clock_gettime_seconds)();
  NFFT(vpr_complex)(p.f_hat, K, "adjoint ndft, vector f_hat (first few entries)");
  printf(" took %.2" NFFT__FES__ " seconds.\n",t1-t0);

  /** approx. adjoint and show the result */
  t0 = NFFT(clock_gettime_seconds)();
  NFFT(adjoint)(&p);
  t1 = NFFT(clock_gettime_seconds)();
  NFFT(vpr_complex)(p.f_hat, K, "adjoint nfft, vector f_hat (first few entries)");
  printf(" took %.2" NFFT__FES__ " seconds.\n",t1-t0);

#endif

  /** finalise the two dimensional plan */
  NFFT(finalize)(&p);
}

int main(void)
{
  printf("1) computing a one dimensional ndft, nfft and an adjoint nfft [dense]\n\n");
  simple_test_nfft_1d(false);

  printf("2) computing a one dimensional ndft, nfft and an adjoint nfft [sparse]\n\n");
  simple_test_nfft_1d(true);
/*
  getc(stdin);

  printf("2) computing a two dimensional ndft, nfft and an adjoint nfft\n\n");
  simple_test_nfft_2d();
*/
  return EXIT_SUCCESS;
}
