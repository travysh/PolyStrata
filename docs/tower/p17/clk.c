#include <stdio.h>
#include <time.h>
#include <x86intrin.h>
int main(void) {
    struct timespec a, b, t;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < 200000; ++i) clock_gettime(CLOCK_MONOTONIC, &t);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ns = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / 200000;
    unsigned long long c0 = __rdtsc(); for (int i = 0; i < 200000; ++i) (void) __rdtsc(); unsigned long long c1 = __rdtsc();
    printf("clock_gettime %.0f ns a call, rdtsc %.1f cycles\n", ns, (c1 - c0) / 200000.0);
    return 0;
}
