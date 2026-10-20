#include <immintrin.h>
#include <stdio.h>
#include <time.h>
#include <x86intrin.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(void){
  for(int r=0;r<7;r++){double t0=now();unsigned long long c0=__rdtsc();
    for(int i=0;i<20000;i++)_mm_pause();
    double t1=now();unsigned long long c1=__rdtsc();
    printf("20000 pause: %.1f us, %.1f tsc/pause\n",t1-t0,(c1-c0)/20000.0);}
  return 0;}
