#include <stdio.h>
#include "../common/book.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 64
#define N 64
#define SIZE (M*1024*1024)

__global__ void myhistKernel(unsigned char * buffer,unsigned int * histo)
{
__shared__ unsigned int temp[256];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

	temp[threadIdx.x]=0;
__syncthreads();

	atomicAdd( &temp[buffer[offset]], 1 );

__syncthreads();
atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

int main( void ) {
    unsigned char *buffer =(unsigned char*)big_random_block( SIZE );
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // allocate memory on the GPU for the file's data
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_buffer, SIZE ) );
    HANDLE_ERROR( cudaMemcpy( dev_buffer, buffer, SIZE,
                              cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_histo,
                              256 * sizeof( int ) ) );
    HANDLE_ERROR( cudaMemset( dev_histo, 0,
                              256 * sizeof( int ) ) );

//dim3 threads(256,256);
dim3 blocks(256,256);
myhistKernel<<<blocks,256>>>(dev_buffer,dev_histo);

    
    unsigned int histo[256];
    HANDLE_ERROR( cudaMemcpy( histo, dev_histo,
                              256 * sizeof( int ),
                              cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
   	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
										start, stop ) );
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );

	long histoCount = 0;
	for (int i=0; i<256; i++) {
		histoCount += histo[i];
	}
	printf( "Histogram Sum:  %ld\n", histoCount );

	// verify that we have the same counts via CPU
	for (int i=0; i<SIZE; i++)
		histo[buffer[i]]--;
	for (int i=0; i<256; i++) {
		if (histo[i] != 0)
			printf( "Failure at %d!\n", i );
	}

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	cudaFree( dev_histo );
	cudaFree( dev_buffer );
	free( buffer );
	getchar();
	cudaThreadExit();
	return 0;
} 