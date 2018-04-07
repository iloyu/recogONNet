#include "countFeatures.cuh"
#define stride 12
#include "SVM.cuh"
#include "opencv2\core\types_c.h"
using namespace std;
 float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG;
     int *d_mask,*d_histo_mask;
    float *device_out,*device_smooth_out,*device_block_out,*device_out_norm,*device_smooth_in;
	int width,Height;

//__global__ void compute_gradients_8UC1_kernel(int height, int width, const PtrElemStep img, 
//                                              float angle_scale, PtrElemStepf grad, PtrElemStep qangle)
//{
//    const int x = blockIdx.x * blockDim.x + threadIdx.x;
//
//    const unsigned char* row = (const unsigned char*)img.ptr(blockIdx.y);
//
//    __shared__ float sh_row[nthreads + 2];
//
//    if (x < width) 
//        sh_row[threadIdx.x + 1] = row[x]; 
//    else 
//        sh_row[threadIdx.x + 1] = row[width - 2];
//
//    if (threadIdx.x == 0)
//        sh_row[0] = row[max(x - 1, 1)];
//
//    if (threadIdx.x == blockDim.x - 1)
//        sh_row[blockDim.x + 1] = row[min(x + 1, width - 2)];
//
//    __syncthreads();
//    if (x < width)
//    {
//        float dx;
//
//        if (correct_gamma)
//            dx = sqrtf(sh_row[threadIdx.x + 2]) - sqrtf(sh_row[threadIdx.x]);
//        else
//            dx = sh_row[threadIdx.x + 2] - sh_row[threadIdx.x];
//
//        float dy = 0.f;
//        if (blockIdx.y > 0 && blockIdx.y < height - 1)
//        {
//            float a = ((const unsigned char*)img.ptr(blockIdx.y + 1))[x];
//            float b = ((const unsigned char*)img.ptr(blockIdx.y - 1))[x];
//            if (correct_gamma)
//                dy = sqrtf(a) - sqrtf(b);
//            else
//                dy = a - b;
//        }
//        float mag = sqrtf(dx * dx + dy * dy);
//
//		float ang = (atan2f(dy, dx) + Pi) * angle_scale - 0.5f;
//        int hidx = (int)floorf(ang);
//        ang -= hidx;
//        hidx = (hidx + cnbins) % cnbins;
//
//        ((uchar2*)qangle.ptr(blockIdx.y))[x] = make_uchar2(hidx, (hidx + 1) % cnbins);
//        ((float2*)  grad.ptr(blockIdx.y))[x] = make_float2(mag * (1.f - ang), mag * ang);
//    }
//}
//void compute_gradients_8UC1(int nbins, int height, int width, const DevMem2D& img, 
//                            float angle_scale, DevMem2Df grad, DevMem2D qangle, bool correct_gamma)
//{
//    const int nthreads = 256;
//
//    dim3 bdim(nthreads, 1);
//    dim3 gdim(divUp(width, bdim.x), divUp(height, bdim.y));
//
//    if (correct_gamma)
//        compute_gradients_8UC1_kernel<nthreads, 1><<<gdim, bdim>>>(height, width, img, angle_scale, grad, qangle);
//    else
//        compute_gradients_8UC1_kernel<nthreads, 0><<<gdim, bdim>>>(height, width, img, angle_scale, grad, qangle);
//
//  checkCudaErrors(cudaGetLastError() );
//
//   checkCudaErrors( cudaDeviceSynchronize() );
//}

__global__ void countCell(float *out,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_histo_mask,int offset_X,int offset_Y)
{
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=blockIdx.y*blockDim.y+threadIdx.y;
	
	int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    
	int off_X=xx+offset_X;
	int off_Y=yy+offset_Y;
	
	__shared__  float histo[1280];//一个圆分18个方向(max(0~17))*方向的宽度70(每个方向7个cell每个cell 10个bin)+扇区编号（max(0~6)）*bin数（10）+属于哪个bin(max(0~9))=17*70+6*10+9=1259
	
	
		/*memset(histo,0,1280*4);*/
	__shared__  float t_fm_nbin[Windowy][Windowx];
	//__shared__  float temp[Windowy][Windowx];
	__shared__  int  t_nm_nbin[Windowy][Windowx];
	
	      __syncthreads();
		  t_fm_nbin[tidy][tidx]=device_p_ANG[off_Y*Imagewidth+off_X]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         t_fm_nbin[tidy][tidx]+=Pi; 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         t_fm_nbin[tidy][tidx]+=Pi; 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		atomicAdd(& (histo[d_histo_mask[yy*m_nImage+xx]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(off_Y)*Imagewidth+ off_X]*d_mask[xx+(yy)*m_nImage]); 
		 /*temp[tidy][tidx]= device_p_Mag[(off_Y)*Imagewidth+ off_X]*d_mask[xx+(yy)*m_nImage];
		 atomicAdd(& (histo[d_histo_mask[yy*m_nImage+xx]+t_nm_nbin[tidy][tidx]]),
			temp[tidy][tidx]);*/
	
		__syncthreads();
		
		atomicAdd(&out[tidy*32+tidx],histo[tidy*32+tidx]);
		
		if(tidy%4==0)
				atomicAdd(&out[1024+(tidy/4)*32+tidx],histo[1024+(tidy/4)*32+tidx]);
		
}

/*
dim3 thread(4,128);一个移动步长为12，需要移动两次
dim3 block(32);
right move
*/
__global__ void right(float *in,int offsetX,int offsetY,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_hist_mask){
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=threadIdx.y;
	int yy_p=yy+offsetY;
	int xx_p=xx+offsetX;
	__shared__  float t_fm_nbin[128][4];
	
	__shared__  int  t_nm_nbin[128][4];
	if(blockIdx.x<3)
	{  t_fm_nbin[yy][threadIdx.x]=device_p_ANG[yy_p*Imagewidth+xx_p]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[yy][threadIdx.x]<0)
         atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
		 if( t_fm_nbin[yy][threadIdx.x]<0)
         atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
		 t_nm_nbin[yy][threadIdx.x]=(int)(t_fm_nbin[yy][threadIdx.x]*10/Pi);
		/* printf(" %d,%d:%d ",yy,threadIdx.x,t_nm_nbin[yy][threadIdx.x]);*/
		 //out[yy*32+threadIdx.x]=t_nm_nbin[yy][threadIdx.x];
		atomicAdd(&in[d_hist_mask[m_nImage*yy+threadIdx.x]+t_nm_nbin[yy][threadIdx.x]],
			-device_p_Mag[(yy_p)*Imagewidth+ xx_p]*d_mask[xx+(yy)*m_nImage]);
	}
__syncthreads();
			if(blockIdx.x>28)
			{   t_fm_nbin[yy][threadIdx.x]=device_p_ANG[yy_p*Imagewidth+xx_p+3]-device_c_ANG[(yy)*m_nImage+xx];
				if( t_fm_nbin[yy][threadIdx.x]<0)
					atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
			if( t_fm_nbin[yy][threadIdx.x]<0)
					atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
			t_nm_nbin[yy][threadIdx.x]=(int)(t_fm_nbin[yy][threadIdx.x]*10/Pi);
	
		 //out[yy*32+threadIdx.x]=t_nm_nbin[yy][threadIdx.x];
			atomicAdd(& (in[d_hist_mask[m_nImage*yy+xx]+t_nm_nbin[yy][threadIdx.x]]),
			device_p_Mag[(yy_p)*Imagewidth+ xx_p+3]*d_mask[xx+(yy)*m_nImage]);
			
			
	}	
			__syncthreads();
			
}
__global__ void left(float *in,float *out,int offsetX,int offsetY,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_hist_mask){
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=threadIdx.y;
	int yy_p=yy+offsetY;
	int xx_p=xx+offsetX;
	__shared__  float t_fm_nbin[128][4];
	
	__shared__  int  t_nm_nbin[128][4];
	if(blockIdx.x<3)
	{  t_fm_nbin[yy][threadIdx.x]=device_p_ANG[yy_p*Imagewidth+xx_p]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[yy][threadIdx.x]<0)
         atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
		 if( t_fm_nbin[yy][threadIdx.x]<0)
         atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
		 t_nm_nbin[yy][threadIdx.x]=(int)(t_fm_nbin[yy][threadIdx.x]*10/Pi);
	
		 //out[yy*32+threadIdx.x]=t_nm_nbin[yy][threadIdx.x];
		atomicAdd(& (in[d_hist_mask[m_nImage*yy+xx]+t_nm_nbin[yy][threadIdx.x]]),
			device_p_Mag[(yy_p)*Imagewidth+ xx_p+3]*d_mask[xx+(yy)*m_nImage]);
	}
			__syncthreads();
			if(blockIdx.x>28)
			{
				 t_fm_nbin[yy][threadIdx.x]=device_p_ANG[yy_p*Imagewidth+xx_p+3]-device_c_ANG[(yy)*m_nImage+xx];
				if( t_fm_nbin[yy][threadIdx.x]<0)
					atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
				if( t_fm_nbin[yy][threadIdx.x]<0)
					atomicAdd(&t_fm_nbin[yy][threadIdx.x],Pi); 
		  
			t_nm_nbin[yy][threadIdx.x]=(int)(t_fm_nbin[yy][threadIdx.x]*10/Pi);
	
		 //out[yy*32+threadIdx.x]=t_nm_nbin[yy][threadIdx.x];
			atomicAdd(& (in[d_hist_mask[m_nImage*yy+xx]+t_nm_nbin[yy][threadIdx.x]]),
			device_p_Mag[(yy_p)*Imagewidth+ xx_p+3]*d_mask[xx+(yy)*m_nImage]);
			
	}	
			
}
/*
dim3 thread(128,4);一个移动步长为12，需要移动两次
dim3 block(32);
right move
*/
__global__ void down(float *in,float *out,int offsetX,int offsetY,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_hist_mask){
	int xx=threadIdx.x;
	int yy=threadIdx.y+blockDim.y*blockIdx.y;
	int yy_p=yy+offsetY;
	int xx_p=xx+offsetX;
	__shared__  float t_fm_nbin[4][128];
	
	__shared__  int  t_nm_nbin[4][128];
	if(blockIdx.y<3)
	{  t_fm_nbin[threadIdx.y][xx]=device_p_ANG[yy_p*Imagewidth+xx_p]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[threadIdx.y][xx]<0)
         atomicAdd(&t_fm_nbin[threadIdx.y][xx],Pi); 
		  
		 if( t_fm_nbin[threadIdx.y][xx]<0)
         atomicAdd(&t_fm_nbin[threadIdx.y][xx],Pi); 
		  
		 t_nm_nbin[threadIdx.y][xx]=(int)(t_fm_nbin[threadIdx.y][xx]*10/Pi);
	
		 //out[threadIdx.y*32+xx]=t_nm_nbin[threadIdx.y][xx];
		atomicAdd(& (in[d_hist_mask[m_nImage*yy+xx]+t_nm_nbin[threadIdx.y][xx]]),
			-device_p_Mag[(yy_p)*Imagewidth+ xx_p]*d_mask[xx+(yy)*m_nImage]);
	}
			__syncthreads();
			if(blockIdx.y>28)
			{   t_fm_nbin[threadIdx.y][xx]=device_p_ANG[yy_p*Imagewidth+xx_p+3]-device_c_ANG[(yy)*m_nImage+xx];
				if( t_fm_nbin[threadIdx.y][xx]<0)
					atomicAdd(&t_fm_nbin[threadIdx.y][xx],Pi); 
		  
			if( t_fm_nbin[threadIdx.y][xx]<0)
					atomicAdd(&t_fm_nbin[threadIdx.y][xx],Pi); 
		  
			t_nm_nbin[threadIdx.y][xx]=(int)(t_fm_nbin[threadIdx.y][xx]*10/Pi);
	
		 //out[threadIdx.y*32+xx]=t_nm_nbin[threadIdx.y][xx];
			atomicAdd(& (in[d_hist_mask[m_nImage*yy+xx]+t_nm_nbin[threadIdx.y][xx]]),
			device_p_Mag[(yy_p)*Imagewidth+ xx_p+3]*d_mask[xx+(yy)*m_nImage]);
			
	}	
			
}
//dim3 block(18,7);//一个cell分18个角度方向,一个方向7个cell，
	//dim3 threads(10);//每个cell 10 个bin
//__global__ void reduce(float *in,float *out){
//	//int id=blockIdx.y*gridDim.x+blockIdx.x;
//	int tidx=threadIdx.x;
//    int tidy=threadIdx.y;
//	int yy=blockIdx.y*blockDim.y+tidy;
//	int xx=blockIdx.x*blockDim.x+tidx;
//	if(blockIdx.x<8) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*8];
//	__syncthreads();
//	if(blockIdx.x<4) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*4];
//	__syncthreads();
//	if(blockIdx.x<2) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*2];
//	__syncthreads();
//
//	if(blockIdx.x==0)out[tidy*blockDim.x+tidx]=in[tidy*blockDim.x+tidx]+in[tidy*blockDim.x+tidx+1280];
//	
////	for(int stride=1280*8;stride>1280;stride/=2)
////	{ if(idx_all<stride)
////	in[idx_all]+=in[idx_all+stride];
////		__syncthreads();
////}
//
//}
__global__ void smoothcell(float *in,float *out){
    int t_nleft,t_nright;
    t_nleft=(threadIdx.x-1+10)%10;
    t_nright=(threadIdx.x+1)%10;
    float *t_ptemp,t_ftemp[10];
    t_ptemp=in+blockIdx.x*70+blockIdx.y*10;//+threadIdx.y)*0.8f+0.1f*(in+blockIdx.x*70+threadIdx.x*10+t_left)
	/*__syncthreads();*/
	if(t_ptemp)
	t_ftemp[threadIdx.x]=t_ptemp[threadIdx.x]*0.8f+0.1f*t_ptemp[t_nleft]+0.1f*t_ptemp[t_nright];
    __syncthreads();
	out[blockIdx.x*70+blockIdx.y*10+threadIdx.x]=t_ftemp[threadIdx.x];
    __syncthreads();
}

//__global__ void smooth(float *in,float *out)
//{
//	int k,j,i;
//	int m_nBIN=10;
//	float *m_pCellFeatures=in;
//	int t_nLineWidth=70;
//	float t_pTemp[10];
//	for ( k = 0; k < 18; ++k )//18
//	{
//		for ( j = 0; j < 7; ++j )//7
//		{
//			for ( i = 0; i< 10; ++i )//10
//			{
//				int t_nLeft;
//				int t_nRight;
//				t_nLeft = ( i - 1 + m_nBIN ) % m_nBIN;
//				t_nRight = ( i + 1 ) % m_nBIN;
//
//				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.8f 
//					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nLeft] * 0.1f 
//					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nRight] * 0.1f;
//			}
//
//			for ( i = 0; i < m_nBIN; ++i )
//			{
//				out[k * t_nLineWidth + j * m_nBIN + i] = t_pTemp[i];
//			}
//		}
//	}
//	
//}

//dim3 block(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
//dim3 threads(3,10);//3=一个block包含3个cell,10=一个cell10个bin
__global__ void countblock(float *in ,float *out)
{
    //if(in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10!=NULL)
   //{ 
	float *ptr_in=in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10;//threadIdx.x;//70=一个角度方向7个cell，每个cell 10个bin,
    float *ptr_out=out+120*blockIdx.x+30*blockIdx.y+10*threadIdx.x;//threadIdx.x;//一个角度方向4个block，一个block3个cell，一个cell 10个bin,
    //一个block3个cell，一个cell 10个bin, 
    ptr_out[threadIdx.y]=ptr_in[threadIdx.y];
	
	////}
    }
     //dim3 block_norm(72);//blob的数量 18*4=72
   // dim3 thread_norm(30);//block特征向量长度（m_nBIN）
__global__ void normalizeL2Hys(float *in,float *out)
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
	sum[bid][0]=sum[bid][0]+sum[bid][1]+sum[bid][2]+sum[bid][3]+sum[bid][4]+sum[bid][5]+sum[bid][6]+sum[bid][7]+sum[bid][8]
	+sum[bid][9]+sum[bid][10]+sum[bid][11]+sum[bid][12]+sum[bid][13]+sum[bid][14];
	/*if(tid<7) sum[bid][tid]+=sum[bid][tid+7];
	 __syncthreads();

	 if(tid<3) sum[bid][tid]+=sum[bid][tid+3];
	 __syncthreads();
	
	  sum[bid][0]=sum[bid][0]+sum[bid][1]+sum[bid][14]+sum[bid][6]+sum[bid][2];*/
	 __syncthreads();
    // Compute the normalization term
	
	 float norm = (rsqrt(sum[bid][0]));
	printf(" %f",norm);
	t_foutemp[tid]=t_ftemp[tid]*norm;
    __syncthreads();


}
template<int size> 
__device__ float reduce_smem(volatile float* smem)
{        
    unsigned int tid = threadIdx.x;
    float sum = smem[tid];

    if (size >= 512) { if (tid < 256) smem[tid] = sum = sum + smem[tid + 256]; __syncthreads(); }
    if (size >= 256) { if (tid < 128) smem[tid] = sum = sum + smem[tid + 128]; __syncthreads(); }
    if (size >= 128) { if (tid < 64) smem[tid] = sum = sum + smem[tid + 64]; __syncthreads(); }
    
    if (tid < 32)
    {        
        if (size >= 64) smem[tid] = sum = sum + smem[tid + 32];
        if (size >= 32) smem[tid] = sum = sum + smem[tid + 16];
        if (size >= 16) smem[tid] = sum = sum + smem[tid + 8];
        if (size >= 8) smem[tid] = sum = sum + smem[tid + 4];
        if (size >= 4) smem[tid] = sum = sum + smem[tid + 2];
        if (size >= 2) smem[tid] = sum = sum + smem[tid + 1];
    }

    __syncthreads();
    sum = smem[0];
    
    return sum;
}

__global__ void normalize_hists_kernel_many_blocks(float *in,float *out)
{
	const float *hist=in+blockDim.x*blockIdx.x;
	 float *hist_norm=out+blockDim.x*blockIdx.x;
	 float product[72]={0};
	int tidx=threadIdx.x;
	__syncthreads();
	product[blockIdx.x]+=hist[tidx]*hist[tidx];
	
	__syncthreads();
	printf(" %f ",product[blockIdx.x]);
	float norm ;
	hist_norm[tidx]=hist[tidx]*rsqrt(product[blockIdx.x]);
}
__host__ void initfeature(float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,
		int ImageHeight,int *mask,int *histo_mask)
{
	
	long size_d_window=sizeof(int)*m_nImage*m_nImage;
    long size_c_window=sizeof(float)*m_nImage*m_nImage;
    long size_c_pixel=sizeof(float)*ImageHeight*Imagewidth;
    long size_s_cell=sizeof(float)*1280;
    long size_c_block=sizeof(float)*2160;
	 checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
	checkCudaErrors(cudaMalloc((void **)&d_mask,size_d_window));
	checkCudaErrors(cudaMalloc((void **)&d_histo_mask,size_d_window));
   
    checkCudaErrors(cudaMalloc((void **)&device_out,size_s_cell));
	checkCudaErrors(cudaMalloc((void **)&device_smooth_in,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));
    checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));

	
    checkCudaErrors(cudaMemcpy(device_c_ANG,c_ANG,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_c_Mag,c_Mag,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_Mag,p_Mag,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_ANG,p_ANG,size_c_pixel,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask,mask,size_d_window,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_histo_mask,histo_mask,size_d_window,cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(device_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_smooth_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_block_out,0,size_c_block));
    checkCudaErrors(cudaMemset(device_out_norm,0,size_c_block));
}
__host__ void CountFeatures(float *out,int off_x,int off_y)
{

	int h_windowx=128/Windowx;
	int h_windowy=128/Windowy;

	dim3 blocks(h_windowx,h_windowy);
	dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量

	dim3 block_right(32);
	dim3 thread_right(4,128);

    dim3 block_smooth(18,7);//一个cell分18个角度方向,一个方向7个cell，
    dim3 threads_smooth(10);//每个cell 10 个bin

	
    dim3 block_b(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
    dim3 thread_b(3,10);//3=一个block包含3个cell,10=一个cell10个bin

    dim3 block_norm(72);//blob的数量 18*4=72
    dim3 thread_norm(30);//block特征向量长度（m_nBIN）

	countCell<<<blocks,threads>>>( device_out, device_c_ANG, device_c_Mag, device_p_ANG, 
		device_p_Mag, Height,width,d_mask,d_histo_mask,off_x,off_y);
	smoothcell<<<block_smooth,threads_smooth>>>(device_out,device_smooth_out);
	countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
	normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);
	
	cudaMemcpy(out,device_out_norm,sizeof(float)*2160,cudaMemcpyDeviceToHost);

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

 extern "C" void countFeaturesfloat(float *out,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,
	 int *mask,int *histo_mask,float SVM_bias,float *svm_weight,int svm_count,int Imagewidth,int ImageHeight,
	 float t_fAugRate,vector <CvRect> &t_vTarget)
{
	float result,*sample;
	width=Imagewidth;
	Height=ImageHeight;
    vector <CvRect> t_vCurRect;
	CvRect t_Rect;
	t_Rect.x = 0;
	t_Rect.y = 0;
	t_Rect.width = m_nImage;
	t_Rect.height = m_nImage;
	initfeature(c_ANG,c_Mag,p_ANG,p_Mag,Imagewidth,ImageHeight,mask,histo_mask);
	InitSVM(SVM_bias,svm_weight,svm_count);
	CountFeatures(out,t_Rect.x,t_Rect.y);
	
	/*while ( t_Rect.x + m_nImage <Imagewidth  )
	{
		t_Rect.y = 0;
		while ( t_Rect.y + m_nImage < ImageHeight )
		{
			CountFeatures(sample,t_Rect.x,t_Rect.y);
			for(int i=0;i<m_nANG;i++)
			{
					LinearSVMEvaluation(svm_weight,sample,SVM_bias,svm_count,result,i);
					if ( result > 0.5f )
				{
					break;
				}
			}

			if( result > 0.5f )
			{
				CvRect t_AddRect;
				t_AddRect.x = (int)( ( t_Rect.x + t_Rect.width * 0.5f ) / t_fAugRate );
				t_AddRect.y = (int)( ( t_Rect.y + t_Rect.height * 0.5f ) / t_fAugRate );
				t_AddRect.width = (int)(  m_nImage/ t_fAugRate ) - 3;
				t_AddRect.height = t_AddRect.width;
				t_AddRect.width /= 2;
				t_AddRect.height /= 2;
				t_vTarget.push_back( t_AddRect );
			}
			t_Rect.y += m_nSearchStep;
		}
		t_Rect.x += m_nSearchStep;
		}
	*/

	Releasefeature();
	CloseSVM();
    cudaDeviceReset();
    
    
}