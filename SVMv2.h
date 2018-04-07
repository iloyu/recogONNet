#include "SVM.cuh"

texture<float, 1, cudaReadModeElementType> texSVM;
cudaArray *svmArray ;
 cudaChannelFormatDesc channelDescSVM=cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
extern __shared__ float allSharedF1[];
float svmBias;
__host__ void InitSVM(float _svmBias, float* svmWeights, int svmWeightsCount)
{
	channelDescSVM = cudaCreateChannelDesc<float>();
	checkCudaErrors(cudaMallocArray(&svmArray, &channelDescSVM, svmWeightsCount, 1));
	checkCudaErrors(cudaMemcpyToArray(svmArray, 0, 0, svmWeights, svmWeightsCount * sizeof(float), cudaMemcpyHostToDevice));
	svmBias = _svmBias;
}
__host__ void CloseSVM()
{
	checkCudaErrors(cudaFreeArray(svmArray));
}
__global__ void linearSVMEvaluation( float result,float svmBias,
									float* hist, 
									int angle)
{
	int tidx=threadIdx.x;;
	int texPos=threadIdx.x+blockDim.x*blockIdx.x;
	int  localPos=threadIdx.x+blockDim.x*((blockIdx.x+angle)%gridDim.x);
	float texValue=tex1D(texSVM,texPos);
	float localval=hist[localPos];
	float* smem = (float*) allSharedF1;
	float product=0;

	
	smem[tidx] =0;

	
	__syncthreads();
	if(tidx<120)
	{
		for(int i=tidx;i<2160;i+=blockDim.x)
		product+=texValue*localval;
		smem[tidx]=product;
	}
	__syncthreads();
	 if (tidx< 64) smem[tidx] = product = product + smem[tidx + 64]; 
        __syncthreads(); 
     if (tidx< 32) smem[tidx] = product = product + smem[tidx + 32]; 
        __syncthreads(); 
		 if (tidx< 16) smem[tidx] = product = product + smem[tidx + 16]; 
        __syncthreads(); 
		 if (tidx< 8) smem[tidx] = product = product + smem[tidx + 8]; 
        __syncthreads(); 
		 if (tidx< 4) smem[tidx] = product = product + smem[tidx + 4]; 
        __syncthreads(); 
		 if (tidx< 2) smem[tidx] = product = product + smem[tidx + 2]; 
        __syncthreads(); 
		if(tidx==0)
			result	=smem[0]+smem[1]-svmBias;
	
}
__host__ void ResetSVMScores(float *svmScores)
{
	checkCudaErrors(cudaMemset(svmScores, 0, sizeof(float)));
}

__host__ void LinearSVMEvaluation(float *coefficients,float * sample,float SVM_bias,int var_count,float score,int angle ){
	
	 const int width=2160; 
    const int height=1; 
	cudaMallocArray(&svmArray,&channelDescSVM,width,height);
	size_t sizeMem=width*height*sizeof(float); 
    size_t potX=0; 
    size_t potY=0; 
	cudaMemcpyToArray(svmArray,potX,potY,coefficients,sizeMem,cudaMemcpyDeviceToHost); 
   
	dim3 threads_SVM=dim3(128,1,1);
	dim3 blocks_SVM=dim3(18,1,1);
	checkCudaErrors(cudaBindTextureToArray(&texSVM,svmArray,&channelDescSVM));
	linearSVMEvaluation<<<blocks_SVM,threads_SVM,sizeof(float)*128>>>(score,SVM_bias,sample,angle);
	checkCudaErrors(cudaUnbindTexture(texSVM));
}