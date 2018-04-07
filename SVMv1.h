#pragma once
#ifndef UTILITIES_H
#define UTILITIES_H
  #endif

int  nextPowerOfTwo(int n)  
{  
    n--;  
    n = n >> 1 | n;  
    n = n >> 2 | n;  
    n = n >> 4 | n;  
    n = n >> 8 | n;  
    n = n >> 16 | n;  
    //n = n >> 32 | n; //For 64-bits int   
   return ++n;
}  
__global__ static void compute_sum(float *array,int cnt , int cnt2)  
{  
    extern __shared__  float sharedMem[];  
    sharedMem[threadIdx.x] = (threadIdx.x < cnt) ? array[threadIdx.x] : 0 ;  
    __syncthreads();  
  
    //cnt2 "must" be a power of two!  
    for( unsigned int s = cnt2/2 ; s > 0 ; s>>=1 )  
    {  
        if( threadIdx.x < s )      
        {  
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];  
        }  
        __syncthreads();  
    }  
    if(threadIdx.x == 0)  
    {  
        array[0] = sharedMem[0];      
    }  
}  
  
  extern "C" void reduceSum(int count,float *array,float *sum){
	   float *deviceArray;  
	   int npt_count;
    cudaMalloc( &deviceArray,count*sizeof(float) );  
    cudaMemcpy( deviceArray,array,count*sizeof(int),cudaMemcpyHostToDevice ); 
	npt_count= nextPowerOfTwo(count);//next power of two of count  
    //cout<<"npt_count = "<<npt_count<<endl;  
    int blockSharedDataSize = npt_count * sizeof(float);  
   for(int i=0;i<count;i++)  
    {  
        compute_sum<<<1,count,blockSharedDataSize>>>(deviceArray,count,npt_count);  
    } 
   cudaMemcpy( &sum,deviceArray,sizeof(int),cudaMemcpyDeviceToHost );  
  }
