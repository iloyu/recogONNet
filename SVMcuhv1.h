#pragma once
#ifndef __SVM_CUH__
#define __SVM_CUH__
////  codes
#include "utilities.cuh"
#include "cuda_texture_types.h"  
__host__ void InitSVM(float _svmBias, float* svmWeights, int svmWeightsCount);
__host__ void CloseSVM();
__global__ void linearSVMEvaluation( float result,float svmBias,
									float* hist, 
									int angle);
__host__ void ResetSVMScores(float *svmScores);
__host__ void LinearSVMEvaluation(float *coefficients,float * sample,float SVM_bias,int var_count,float score,int angle );
	
#endif
//

