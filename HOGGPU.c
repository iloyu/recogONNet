#include "countFeatures.cuh"
#include "opencv2\core\cuda_devptrs.hpp"
#include "cublas_v2.h"
#define stride m_nSearchStep
#define Num  10000
#define BIAS 3.3912737032743312e+000
#include "opencv2\core\types_c.h"
#include "SVM.cuh"
#include "cublas_v2.h"
#include "common\book.h"
#include "cuda_runtime.h"
#include "texture_indirect_functions.h"
using namespace cv;
using namespace gpu;

texture<float, 2, cudaReadModeElementType> texGrad;
texture<float, 2, cudaReadModeElementType> texAng;

texture<float, 2, cudaReadModeElementType> texImg;

cudaArray *svmArray2 ;
cudaArray *GradArray;
cudaArray *AngArray;
cudaArray *CGradArray;
cudaArray *CAngArray;
cudaArray *CHisto;
cudaArray *CMask;

cublasHandle_t handle;
  


float *result_tmp,*d_svm,*d_sample;

size_t pitch;
 
int *result;
int *num;


using namespace std;
 float *d_img;
 //uchar *d_resize;
float *d_resize;  
 float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG;
 int *d_mask,*d_histo_mask;

float *device_out,*device_smooth_out,*device_block_out,*device_out_norm;
float **device_out3;
//int width,Height;
	
	typedef struct Rect
{
	int x;
	int y;
	int width;
	int height;
}
myRect;
myRect *t_Target;
__global__ void smode(float *m_fang,float *m_fmag,int *hist,int *mask)
{
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=blockIdx.y*blockDim.y+threadIdx.y;
	int d_xx=xx-m_halfWindow;
	int d_yy=m_halfWindow-yy;
	__shared__ int  t_mnAng[32*32];
	__shared__ int  t_mnMag[32*32];
	
	m_fang[yy*m_window+xx]=atan2f(d_yy,d_xx);
	if(m_fang[yy*m_window+xx]<0)
		m_fang[yy*m_window+xx]+=PI2;

	m_fmag[yy*m_window+xx]=sqrtf(d_yy*d_yy+d_xx*d_xx);
	__syncthreads();
	t_mnAng[threadIdx.y*blockDim.x+threadIdx.x]=(int)(m_fang[yy*m_window+xx]*9/Pi);
	t_mnMag[threadIdx.y*blockDim.x+threadIdx.x]=(int)(m_fmag[yy*m_window+xx]/10);
	hist[yy*m_window+xx]=t_mnAng[threadIdx.y*blockDim.x+threadIdx.x]*70+10*t_mnMag[threadIdx.y*blockDim.x+threadIdx.x];
	__syncthreads();
	if(m_fmag[yy*m_window+xx]<m_halfWindow)
		mask[yy*m_window+xx]=1;
	else mask[yy*m_window+xx]=0;
			
}

__host__ void gpu_hog(float* src,int width,int height)
{
	long size_d_window=sizeof(int)*m_nImage*m_nImage;
	long size_c_window=sizeof(float)*m_nImage*m_nImage;
	long size_c_pixel=sizeof(float)*height*width;
	checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
	checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));

	checkCudaErrors(cudaMalloc((void **)&d_mask,size_d_window));
	checkCudaErrors(cudaMalloc((void **)&d_histo_mask,size_d_window));

	dim3 block=dim3(H_window/32,H_window/32);
	dim3 thread=dim3(32,32);
	//求模版参数
	smode<<<block,thread>>>(device_c_ANG,device_c_Mag,d_histo_mask,d_mask);

	int size=H_window*H_window;
	
	cudaChannelFormatDesc channelDescImg =    cudaCreateChannelDesc<float>();
	 cudaArray* ImgArray;
	cudaMallocArray(&ImgArray, &channelDescImg, width, height);
	cudaMemcpyToArray(ImgArray, 0, 0, src, width*height*sizeof(float), cudaMemcpyHostToDevice);
	 texImg.addressMode[0] = cudaAddressModeClamp;   //寻址模式规定了纹理拾取的输入坐标超出纹理寻址范围时的行为
	texImg.addressMode[1] = cudaAddressModeClamp;   //clamp:超出部分按寻址范围的最大值或最小值处理；wrap:只针对归一化，超出部分整数部分去掉，取小数部分，即循环处理；
	texImg.filterMode = cudaFilterModeLinear;       //线性化或者取最近的像素点值。线性化对一维就是线性，对二维是双线性。
	texImg.normalized = false;                      //此处调整后tex2D中源图像坐标也要调整是否归一化？
	 cudaBindTextureToArray(&texImg, ImgArray, &channelDescImg);
 
	int size_s_cell=sizeof(float)*1280;
	int size_c_block=sizeof(float)*2160;
	checkCudaErrors(cudaMalloc((void **)&device_out,size_s_cell));
	checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
	checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));
	checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));

}

int divUp(int x,int y)
{
	return (x+y-1)/y;
}
__global__ void resize_GPU(int dstx, int dsty, float* dst,int width,int Height,float *src,float scale)
{
	 int x = blockIdx.x * blockDim.x + threadIdx.x;
	 int y = blockIdx.y * blockDim.y + threadIdx.y;
	double srcXf;  
	double srcYf;  
	int srcX;  
	int srcY;  
	double u;  
	double v;  
	long dstOffset;  
	if(x>=dstx || y>=dsty) return;  
	srcXf=  x/scale;//* ((float)width/dstx);//这个float重要阿，不然会算是精度的，尼玛  
	srcYf =  y/scale;//*((float)Height/dsty);  
	
	srcX = (int)srcXf;  
	srcY = (int)srcYf;  
	u= srcXf - srcX;  
	v = srcYf - srcY;          
	dstOffset =(y*dstx+x);  
	dst[dstOffset] = 0;  
	dst[dstOffset]+=(1-u)*(1-v)*src[(srcY*width+srcX)];  
	dst[dstOffset]+=(1-u)*v*src[((srcY+1)*width+srcX)];  
	dst[dstOffset]+=u*(1-v)*src[(srcY*width+srcX+1)];  
	dst[dstOffset]+= u*v*src[((srcY+1)*width+srcX+1)];  
}  
__global__ void resize_GPU2(int dstx, int dsty, float* dst,int width,int Height,float scale)
{
	 int x = blockIdx.x * blockDim.x + threadIdx.x;
	 int y = blockIdx.y * blockDim.y + threadIdx.y;
	double srcXf;  
	double srcYf;  


	long dstOffset;  
	if(x>=dstx || y>=dsty) return;  
	srcXf=  x/scale;//+0.5;//* ((float)width/dstx);//这个float重要阿，不然会算是精度的，尼玛  
	srcYf =  y/scale;//+0.5;//*((float)Height/dsty);  
	//if(scale>1) {srcXf=x*scale;srcYf=y*scale;}

   /* srcX = (int)srcXf;  
	srcY = (int)srcYf;  
	u= srcXf - srcX;  
	v = srcYf - srcY; */         
	dstOffset =(y*dstx+x);  
 
	dst[dstOffset]=tex2D(texImg,srcXf,srcYf);
   
}  

__host__ void resize_for_hog(int dsty, int dstx,int width,int height,float scale)
{   
	int uint = 16;  
	long size_c_pixel=sizeof(float)*dstx*dsty;
	dim3 grid((dstx+uint-1)/uint,(dsty+uint-1)/uint);  
	dim3 block(uint,uint);  
	HANDLE_ERROR(cudaMalloc((void**)&d_resize,dstx*dsty*sizeof(float)));  
   /* if(1-scale<0.0001)
	d_resize=d_img;
	else*/
	//resize_GPU<<<grid,block>>>(dstx, dsty,d_resize,width,height,d_img,scale);  
	resize_GPU2<<<grid,block>>>(dstx,dsty,d_resize,width,height,scale);
	checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
	checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
	
}

/*
初始化在gpu上进行运算的中间变量
分配内存
*/

__global__ void compute_gradients_8UC1_kernel(int height, int width,  float* img, 
											   float* grad, float* qangle)
{
	 int x = blockIdx.x * blockDim.x + threadIdx.x;

	 float* row = (img+blockIdx.y*width);

	 __shared__ float sh_row[256+2];

	if (x < width) 
		sh_row[threadIdx.x+1] = row[x]; 
	else 
		sh_row[threadIdx.x + 1] = row[width - 2];

	if (threadIdx.x == 0)
		sh_row[0] = row[max(x - 1, 1)];

	if (threadIdx.x == blockDim.x - 1)
		sh_row[blockDim.x + 1] = row[min(x + 1, width - 2)];

	__syncthreads();
	if (x < width)
	{   float  dx;

		dx=sh_row[threadIdx.x + 2]-sh_row[threadIdx.x ]; ////sqrtf(sh_row[threadIdx.x + 1]) - sqrtf(sh_row[threadIdx.x-1]);
		float dy = 0.f;
		if (blockIdx.y > 0 && blockIdx.y < height - 1)
		{
		
			float a=(img+(blockIdx.y+1)*width)[x];
			float b=(img+(blockIdx.y-1)*width)[x];
			
				dy = a-b;//sqrtf(a) - sqrtf(b);//a-b;//
		}
		float mag = sqrtf(dx * dx + dy * dy);
		float ang = atan2f(dx, dy) ;
		if(ang<0) ang+=Pi;
		if(blockIdx.y==0||blockIdx.y==height-1) return;
		(qangle+blockIdx.y*width)[x] = ang;
		(grad+blockIdx.y*width)[x] = mag ;
	}
}
__host__ void InitSVM( float* svmWeights, int svmWeightsCount)
{

	checkCudaErrors(cudaMalloc((void **)&d_svm,svmWeightsCount*18*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_svm,svmWeights,svmWeightsCount*18*sizeof(float),cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void **)&result_tmp,sizeof(float)*18));
	checkCudaErrors(cudaMalloc((void **)&result,sizeof(int)));

	checkCudaErrors(cudaMalloc((void **)&t_Target,sizeof(myRect)*Num));
	checkCudaErrors(cudaMalloc((void **)&num,sizeof(int)));
	/*cudaMemset(num,0,sizeof(int));*/
}
__host__ void compute_gradients_8UC1( int height, int width,  float * img, 
							  float* grad, float* qangle)
{
	const int nthreads = 256;
	dim3 bdim(nthreads, 1);
	dim3 gdim((width+bdim.x-1)/bdim.x,height/bdim.y);// divUp()
	
   compute_gradients_8UC1_kernel<<<gdim, bdim>>>(height, width, img,  grad, qangle);
  const cudaChannelFormatDesc  channelDescGrad = cudaCreateChannelDesc<float>();
  const cudaChannelFormatDesc channelDescAng = cudaCreateChannelDesc<float>();
	int size=width*height;
	
   checkCudaErrors(cudaMallocArray(&GradArray,&channelDescGrad,width,height));
   checkCudaErrors(cudaMemcpyToArray(GradArray, 0, 0, grad, size * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaBindTextureToArray(&texGrad,GradArray,&channelDescGrad));
	checkCudaErrors(cudaMallocArray(&AngArray, &channelDescAng, width, height));
	checkCudaErrors(cudaMemcpyToArray(AngArray, 0, 0, qangle, size * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaBindTextureToArray(&texAng,AngArray,&channelDescAng));
}

__global__ void countCell(float *out,float *c_ANG,float *p_ANG,
						  float *p_Mag,int ImageHeight,int Imagewidth,
						  int *mask,int *histo_mask,int offset_X,int offset_Y)
{
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=blockIdx.y*blockDim.y+threadIdx.y;
	
	int tidx=threadIdx.x;
	int tidy=threadIdx.y;
	
	int off_X=xx+offset_X;
	int off_Y=yy+offset_Y;
	
	__shared__  float histo[1280];
	//一个圆分18个方向(max(0~17))*方向的宽度70(每个方向7个cell每个cell 10个bin)+
	//扇区编号（max(0~6)）*bin数（10）+属于哪个bin(max(0~9))=17*70+6*10+9=1259
	//memset(histo,0,1280*sizeof(float));
	histo[tidx*blockDim.y+tidy]=0;
	out[tidx*blockDim.y+tidy]=0;
	if(tidx%4==0)
		{histo[1024+(tidx/4)*blockDim.y+tidy]=0;
	out[1024+(tidx/4)*blockDim.y+tidy]=0;}
	__syncthreads();
	__shared__  float t_fm_nbin[Windowy][Windowx];
	
	__shared__  int  t_nm_nbin[Windowy][Windowx];
	
		  __syncthreads();
		  t_fm_nbin[tidy][tidx]=p_ANG[off_Y*Imagewidth+off_X]-c_ANG[(yy)*m_nImage+xx];
		 
		if( t_fm_nbin[tidy][tidx]<0)
		 t_fm_nbin[tidy][tidx]+=Pi; 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
		 t_fm_nbin[tidy][tidx]+=Pi; 
		 
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	/*  printf("%f ",t_fm_nbin[tidy][tidx]);*/
		 __syncthreads();
		 
		atomicAdd(& (histo[histo_mask[yy*m_nImage+xx]+t_nm_nbin[tidy][tidx]]),
			p_Mag[(off_Y)*Imagewidth+ off_X]*mask[xx+(yy)*m_nImage]); 
	
		__syncthreads();
		
		atomicAdd(&out[tidy*32+tidx],histo[tidy*32+tidx]);
		
		if(tidy%4==0)
				atomicAdd(&out[1024+(tidy/4)*32+tidx],histo[1024+(tidy/4)*32+tidx]);
		__syncthreads();	
		
}



__global__ void smoothcell(float *in,float *out){

	float *t_ptemp;t_ptemp=in+blockIdx.x*70+blockIdx.y*10;
	__shared__ float t_ftemp[10+2];
	t_ftemp[threadIdx.x+1]=t_ptemp[threadIdx.x];
	t_ftemp[0]=t_ptemp[9];
	t_ftemp[11]=t_ptemp[0];
	__syncthreads();
	out[blockIdx.x*70+blockIdx.y*10+threadIdx.x]=t_ftemp[threadIdx.x+1]*0.8f+0.1f*t_ptemp[threadIdx.x]+0.1f*t_ptemp[threadIdx.x+2];

}

//dim3 block(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
//dim3 threads(3,10);//3=一个block包含3个cell,10=一个cell10个bin
__global__ void countblock(float *in ,float *out)
{
	
	float *ptr_in=in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10;//threadIdx.x;//70=一个角度方向7个cell，每个cell 10个bin,
	float *ptr_out=out+120*blockIdx.x+30*blockIdx.y+10*threadIdx.x;//threadIdx.x;//一个角度方向4个block，一个block3个cell，一个cell 10个bin,
	//一个block3个cell，一个cell 10个bin, 
	ptr_out[threadIdx.y]=ptr_in[threadIdx.y];
	
}
	 //dim3 block_norm(72);//blob的数量 18*4=72
   // dim3 thread_norm(30);//block特征向量长度（m_nBIN）
__global__ void normalizeL2norm(float *in,float *out)
{
	int bid=blockIdx.x;
	int tid=threadIdx.x;
	// Sum the vector
	__shared__ float sum[72][15];//15*72
   //memset(sum[15],0,15*sizeof(float));
   __syncthreads();
	float *t_ftemp=in+bid*30;
	float *t_foutemp=out+bid*30;
	if(tid<15) sum[bid][tid]=t_ftemp[tid+15]*t_ftemp[tid+15]+t_ftemp[tid]*t_ftemp[tid];
	__syncthreads();
	if(tid<7) sum[bid][tid]+=sum[bid][tid+7];
	__syncthreads();
	 if(tid<3) sum[bid][tid]+=sum[bid][tid+3];
	 __syncthreads();
	
	  sum[bid][0]=sum[bid][0]+sum[bid][1]+sum[bid][14]+sum[bid][6]+sum[bid][2];
	__syncthreads();
	
	if(sum[bid][0]<=0)
		sum[bid][0]=1;
	 float norm = (rsqrt(sum[bid][0]));
	
	t_foutemp[tid]=t_ftemp[tid]*norm;
	//__syncthreads();
	
}

__host__ void CountFeatures(int off_x,int off_y,int im_height,int im_width)
{

	int h_windowx=H_window/Windowx;
	int h_windowy=H_window/Windowy;
	/*int h_windowz=(im_width-H_window)/stride;*/

	dim3 blocks(h_windowx,h_windowy);
	dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量

	dim3 block_right(32);
	dim3 thread_right(4,H_window);

	dim3 block_smooth(18,7);//一个cell分18个角度方向,一个方向7个cell，
	dim3 threads_smooth(10);//每个cell 10 个bin

	
	dim3 block_b(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
	dim3 thread_b(3,10);//3=一个block包含3个cell,10=一个cell10个bin

	dim3 block_norm(72);//blob的数量 18*4=72
	dim3 thread_norm(30);//block特征向量长度（m_nBIN）

	//countCell3<<<blocks,threads>>>(device_out3,device_c_ANG,device_p_ANG,device_p_Mag,im_height,im_width,d_mask,d_histo_mask,off_y);
	countCell<<<blocks,threads>>>(device_out,device_c_ANG,device_p_ANG,device_p_Mag,im_height,im_width,d_mask,d_histo_mask,off_x,off_y);
	//countCell2<<<blocks,threads>>>( device_out,device_c_ANG,d_mask,im_height,im_width,off_x,off_y );

	smoothcell<<<block_smooth,threads_smooth>>>(device_out,device_smooth_out);

	countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);

	normalizeL2norm<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);

	

}
__host__ void Releasefeature()
{
	cudaFree(device_c_ANG);
	cudaFree(device_c_Mag);
	cudaFree(device_p_ANG);
	cudaFree(device_p_Mag);
   
	cudaFree(device_out);
	cudaFree(device_smooth_out);
	cudaFree(device_block_out);
	cudaFree(device_out_norm);
	
}



__global__ void linearSVMEvaluation2( float* hist,float *score,float* svm,int *precision,float t_fStartRate,myRect t_Rect,myRect *t_vTarget,int* num)
{
	//if(threadIdx.x==0&&blockIdx.x==0)
	//	score[angle]=0;
	//int texPos=threadIdx.x+blockDim.x*((blockIdx.x+blockIdx.y)%18);
	score[blockIdx.y]=0;
	__syncthreads();
	int texPos=threadIdx.x+blockDim.x*(blockIdx.x);
	//float texValue=tex1D(texSVM,texPos);
	float texValue=svm[texPos+blockIdx.y*2160];//tex2D(texSVM2,);
	double tmp[18];
	//score[blockIdx.y]=0;
	//__syncthreads();
	//printf("%d %d %f \n",texPos,blockIdx.y,texValue);
	tmp[blockIdx.y]=0;
		int  localPos=threadIdx.x+blockDim.x*blockIdx.x;
		//int  localPos=threadIdx.x+blockDim.x*(blockIdx.x);
	float localval=hist[localPos];
		tmp[blockIdx.y]+=texValue*localval;
   __syncthreads();
   atomicAdd(&score[blockIdx.y],tmp[blockIdx.y]);
   __syncthreads();
   precision[0]=0;
   __syncthreads();
	if(score[blockIdx.y]+BIAS<0)
		precision[0]=1;
	__syncthreads();
	if(precision[0])
	{
		myRect t_AddRect;
				t_AddRect.x = (int)( ( t_Rect.x + t_Rect.width * 0.5f ) / t_fStartRate );
				t_AddRect.y = (int)( ( t_Rect.y + t_Rect.height * 0.5f ) / t_fStartRate );
				t_AddRect.width = (int)(  m_nImage/ t_fStartRate ) - 3;
				t_AddRect.height = t_AddRect.width;
				t_AddRect.width /= 2;
				t_AddRect.height /= 2;
				t_vTarget[num[0]] =t_AddRect ;
				precision[0]=0;
				num[0]++;
	}


}



__host__ void LinearSVMEvaluation(float * sample,float t_fStartRate,myRect t_Rect,float *svm)
{
  

	dim3 blocks=dim3(18,18);

	linearSVMEvaluation2<<<blocks,120>>>(sample,result_tmp,svm,result,t_fStartRate,t_Rect,t_Target, num);
	
}
__host__ void CloseSVM()
{

	checkCudaErrors(cudaFree(result));
	checkCudaErrors(cudaFree(d_sample));
	checkCudaErrors(cudaFree(d_svm));
	//cublasDestroy(handle);
}


 extern "C" void countFeaturesfloat(float *img,float SVM_bias,float *svm_weight,int svm_count,
	 int Imagewidth,int ImageHeight,
	float t_fStartRate ,float t_fStepSize,int t_nResizeStep,vector <CvRect> &t_vTarget)
{
	cudaSetDevice(0);  
	cudaFree(0);   
	//float res[18];
	int i,j;
	//int *precision=new int[1];//=0;
	
	vector <myRect> t_vCurRect;
	myRect t_Rect;
	t_Rect.x = 0;
	t_Rect.y = 0;
	t_Rect.width = m_nImage;
	t_Rect.height = m_nImage;
	FILE *fp;
	
	InitSVM(svm_weight,svm_count);
	gpu_hog(img,Imagewidth,ImageHeight);
	myRect* precision=new myRect[Num];
	for(i=0;i<t_nResizeStep;++i)
	{
		int t_nNewWidth;
		int t_nNewHeight;
		t_nNewHeight=(int)ImageHeight*t_fStartRate;
		t_nNewWidth=(int)Imagewidth*t_fStartRate;
		resize_for_hog(t_nNewHeight,t_nNewWidth,Imagewidth,ImageHeight,t_fStartRate);
		compute_gradients_8UC1(t_nNewHeight,t_nNewWidth,d_resize,device_p_Mag,device_p_ANG);
		t_Rect.y=0;

		while ( t_Rect.y + m_nImage < t_nNewHeight )
		{
				  t_Rect.x = 0;
			while ( t_Rect.x + m_nImage <t_nNewWidth  )
			{
				CountFeatures(t_Rect.x,t_Rect.y,t_nNewHeight,t_nNewWidth);
				LinearSVMEvaluation(device_out_norm,  t_fStartRate,t_Rect,d_svm);
				t_Rect.x += m_SearchStep;
		}
		t_Rect.y += m_SearchStep;
	}
		//ccall(t_nNewWidth,t_nNewHeight);
	t_fStartRate += t_fStepSize;		//更新放大率

	}
	int h_num;
	checkCudaErrors(cudaMemcpy(&h_num,num,sizeof(int),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(precision,t_Target,sizeof(myRect)*h_num,cudaMemcpyDeviceToHost));
		for(int i=0;i<h_num;i++)
	{
		CvRect t_AddRect;

		t_AddRect.x=precision[i].x;
		t_AddRect.y=precision[i].y;
		t_AddRect.width=precision[i].width;
		t_AddRect.height=precision[i].height;
		t_vTarget.push_back(t_AddRect);
	}
 }
	
	
	
	
