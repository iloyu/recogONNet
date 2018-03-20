/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include <stdio.h>
#include "../common/book.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZE    (60*1024*1024)


__global__ void histo_kernel( unsigned char *buffer,
							  long size,
							  unsigned int *histo ) {
								  __shared__   int temp[64];
								   int x=blockDim.x*blockIdx.x+threadIdx.x;
								  int y=blockDim.y*blockIdx.y+threadIdx.y;
								  int tid=x+y*gridDim.x*blockDim.x;
								  //printf("offset:%ld",gridDim.x*blockDim.x);
								  temp[threadIdx.x]=0;
								 int offset=gridDim.x*blockDim.x;
								  __syncthreads();
								  while(tid<size)
								  {
									  int val=buffer[tid];
									   atomicAdd(&temp[val],1);
									   tid+=offset;
									   //offset+=gridDim.x*blockDim.x;
								  }
								 
								  __syncthreads();
								  atomicAdd(&(histo[threadIdx.x]),temp[threadIdx.x]);
}

int main( void ) {
	unsigned char *buffer =
					 (unsigned char*)big_random_block( SIZE );

	// capture the start time
	// starting the timer here so that we include the cost of
	// all of the operations on the GPU.  if the data were
	// already on the GPU and we just timed the kernel
	// the timing would drop from 74 ms to 15 ms.  Very fast.
	cudaEvent_t     start, stop;
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
							  64 * sizeof( int ) ) );
	HANDLE_ERROR( cudaMemset( dev_histo, 0,
							  64 * sizeof( int ) ) );

	// kernel launch - 2x the number of mps gave best timing
	cudaDeviceProp  prop;
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
	int blocks = prop.multiProcessorCount;
	printf("blocks:%d",blocks);
	histo_kernel<<<blocks*2,64>>>( dev_buffer,
									SIZE, dev_histo );
	
	unsigned int    histo[64];
	HANDLE_ERROR( cudaMemcpy( histo, dev_histo,
							  64 * sizeof( int ),
							  cudaMemcpyDeviceToHost ) );

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
										start, stop ) );
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );

	long histoCount = 0;
	for (int i=0; i<64; i++) {
		histoCount += histo[i];
	}
	printf( "Histogram Sum:  %ld\n", histoCount );

	// verify that we have the same counts via CPU
	for (int i=0; i<SIZE; i++)
		histo[buffer[i]]--;
	for (int i=0; i<64; i++) {
		if (histo[i] != 0)
			printf( "Failure at %d!\n", i );
	}

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
	cudaFree( dev_histo );
	cudaFree( dev_buffer );
	free( buffer );
	getchar();

	return 0;
}
