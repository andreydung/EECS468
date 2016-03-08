#include <time.h>

#define TIMER_INIT timespec start_time, end_time
#define START_TIMER clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time)
#define END_TIMER clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time)
#define TIMER_DIFF elapsed_s( start_time, end_time )

float elapsed_s( timespec start, timespec end );
