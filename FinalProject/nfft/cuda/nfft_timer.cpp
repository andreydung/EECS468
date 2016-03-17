#include <time.h>
#include "nfft_timer.h"

float elapsed_s( timespec start, timespec end )
{
        float elapsed = end.tv_nsec - start.tv_nsec;
        elapsed = elapsed * 1.0E-9 + end.tv_sec - start.tv_sec;
        return elapsed;
}

