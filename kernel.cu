
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"iostream"
#include <stdio.h>
#include "../common/book.h"
#include"time.h"
#define SIZE    (100*1024*1024)

int main( void ) {
    unsigned char *buffer =
                     (unsigned char*)big_random_block( SIZE );

    // capture the start time
    clock_t         start, stop;
    start = clock();

    unsigned int    histo[256];
    for (int i=0; i<256; i++)
        histo[i] = 0;

    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]++;

    stop = clock();
    float   elapsedTime = (float)(stop - start) /
                          (float)CLOCKS_PER_SEC * 1000.0f;
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    long histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld\n", histoCount );

    free( buffer );
	getchar();
    return 0;
}
