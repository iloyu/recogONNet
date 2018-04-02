#include "countFeatures.cuh"
#define stride 12

 /*texture<float,cudaTextureType2D, cudaReadModeElementType> t_p_Mag;
 texture<float,cudaTextureType2D, cudaReadModeElementType> t_p_ANG;*device_d_ANG,*device_d_Mag,*/
    //long h_windowx=Imagewidth/Windowx;
    //long h_windowy=ImageHeight/Windowy;
    //dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
    //dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量

//__global__ void countCell(float *out,int *device_d_ANG,int *device_d_Mag,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_histo_mask)
////{
__global__ void countCell(float *out,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_histo_mask)
//{
//__global__ void countCell(float *out,int ImageHeight,int Imagewidth)
{
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=blockIdx.y*blockDim.y+threadIdx.y;
	
	int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    
	int id=blockIdx.x+blockIdx.y*blockDim.x;
	
	
	__shared__  float histo[1280];//一个圆分18个方向(max(0~17))*方向的宽度70(每个方向7个cell每个cell 10个bin)+扇区编号（max(0~6)）*bin数（10）+属于哪个bin(max(0~9))=17*70+6*10+9=1259
	memset(histo,0,1280*4);
	__shared__  float t_fm_nbin[32][32];
	
	__shared__  int  t_nm_nbin[32][32];
	
	      __syncthreads();
		  t_fm_nbin[tidy][tidx]=device_p_ANG[yy*Imagewidth+xx]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		/*atomicAdd(& (histo[device_d_ANG[(yy)*m_nImage+xx]*70+device_d_Mag[(yy)*m_nImage+xx]*10+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);*/
		 atomicAdd(& (histo[d_histo_mask[yy*m_nImage+xx]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);
		
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

	if(tid<7) sum[bid][tid]+=sum[bid][tid+7];
	 __syncthreads();

	 if(tid<3) sum[bid][tid]+=sum[bid][tid+3];
	 __syncthreads();
	/* if(tid<2) sum[bid][tid]+=sum[bid][tid+2];
	 __syncthreads();*/
	 if(tid==0) sum[bid][tid]=sum[bid][tid]+sum[bid][tid+1]+sum[bid][14]+sum[bid][6]+sum[bid][2];
	 __syncthreads();
    // Compute the normalization term
	
	 float norm = (rsqrt(sum[bid][0]));
	/*if(sum[1]-0<0.000001) norm=0;*/
	 //printf(" %f ",sum[bid][0]);
	//printf(" %f,%f ",sum[7],norm);
	t_foutemp[tid]=t_ftemp[tid]*norm;
    __syncthreads();


}


 extern "C" void countFeaturesfloat(float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask,int *histo_mask)
{
	 float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG;
 int *d_mask,*d_histo_mask;
    //int *device_d_ANG,*device_d_Mag,*d_mask;
    //float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG,*device_out,*device_smooth_out,*device_block_out,*device_out_norm,*device_smooth_in;
    float *device_out,*device_smooth_out,*device_block_out,*device_out_norm,*device_smooth_in;
	//uchar *device_in;
	//void * m_pClassifier;//分类器指针
	//float t_nRes;//SVM分类后的概率
	//CvMat *t_FeatureMat;
	//CvSVM * t_pSVM = new CvSVM;
	//	t_pSVM->load( "C:\\Users\\Cyj\\Desktop\\123.xml" );
	//	/*m_pClassifier = (void *)t_pSVM;*/
	//t_FeatureMat = cvCreateMat(  1, 2160,CV_32FC1 );

    long size_d_window=sizeof(int)*m_nImage*m_nImage;
    long size_c_window=sizeof(float)*m_nImage*m_nImage;
    long size_c_pixel=sizeof(float)*ImageHeight*Imagewidth;
    long size_uc_pixel=sizeof(unsigned char)*ImageHeight*Imagewidth;
    long size_c_cell=sizeof(float)*1280*(ImageHeight/m_nImage)*(Imagewidth/m_nImage);
    long size_s_cell=sizeof(float)*1280;
    long size_c_block=sizeof(float)*2160;

    checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));
   /* checkCudaErrors(cudaMalloc((void **)&device_d_ANG,size_d_window));
    checkCudaErrors(cudaMalloc((void **)&device_d_Mag,size_d_window));*/
    checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
	checkCudaErrors(cudaMalloc((void **)&d_mask,size_d_window));
	checkCudaErrors(cudaMalloc((void **)&d_histo_mask,size_d_window));
   
    checkCudaErrors(cudaMalloc((void **)&device_out,size_c_cell));
	checkCudaErrors(cudaMalloc((void **)&device_smooth_in,size_c_cell));
    checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));
    checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));
	/* checkCudaErrors(cudaMalloc((void **)&device_in,size_uc_pixel));*/

    checkCudaErrors(cudaMemcpy(device_c_ANG,c_ANG,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_c_Mag,c_Mag,size_c_window,cudaMemcpyHostToDevice));
    /*checkCudaErrors(cudaMemcpy(device_d_Mag,d_Mag,size_d_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_d_ANG,d_ANG,size_d_window,cudaMemcpyHostToDevice));*/
    checkCudaErrors(cudaMemcpy(device_p_Mag,p_Mag,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_ANG,p_ANG,size_c_pixel,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask,mask,size_d_window,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_histo_mask,histo_mask,size_d_window,cudaMemcpyHostToDevice));
	/*checkCudaErrors(cudaMemcpyToSymbol(device_c_ANG,&c_ANG,sizeof(c_ANG)));
	checkCudaErrors(cudaMemcpyToSymbol(device_c_Mag,&c_Mag,sizeof(c_Mag)));*/
	/*checkCudaErrors(cudaMemcpyToSymbol(device_d_Mag,&d_Mag,sizeof(d_Mag)));
	checkCudaErrors(cudaMemcpyToSymbol(device_d_ANG,&d_ANG,sizeof(d_ANG)));*/
	/*checkCudaErrors(cudaMemcpyToSymbol(device_p_Mag,&p_Mag,sizeof(p_Mag)));
	checkCudaErrors(cudaMemcpyToSymbol(device_p_ANG,&p_ANG,sizeof(p_ANG)));*/
	
	//checkCudaErrors(cudaMemcpyToSymbol(d_mask,&mask,sizeof(d_mask)));
	  // Allocate CUDA array in device memory    
	//const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);   
	//cudaArray* cuArray_Mag,*cuArray_ANG;    
	//cudaMallocArray(&cuArray_Mag, &channelDesc, Imagewidth, ImageHeight);  
	//cudaMallocArray(&cuArray_ANG, &channelDesc, Imagewidth, ImageHeight);  
	//// Copy to device memory some data located at address h_data   
	//// in host memory    
	//cudaMemcpyToArray(cuArray_Mag, 0, 0, p_Mag, sizeof(p_Mag), cudaMemcpyHostToDevice);  
	//cudaMemcpyToArray(cuArray_ANG, 0, 0, p_ANG, sizeof(p_ANG), cudaMemcpyHostToDevice);  
	//// Set texture reference parameters   
	//t_p_Mag.addressMode[0] = cudaAddressModeWrap;   
	//t_p_Mag.addressMode[1] = cudaAddressModeWrap;   
	//t_p_Mag.filterMode     = cudaFilterModeLinear;   
	//t_p_Mag.normalized     = true;   
	//t_p_ANG.addressMode[0] = cudaAddressModeWrap;   
	//t_p_ANG.addressMode[1] = cudaAddressModeWrap;   
	//t_p_ANG.filterMode     = cudaFilterModeLinear;   
	//t_p_ANG.normalized     = true;
	//// Bind the array to the texture reference   
	//cudaBindTextureToArray(t_p_Mag, cuArray_Mag,channelDesc); 
	//cudaBindTextureToArray(t_p_ANG, cuArray_ANG,channelDesc);
 //   checkCudaErrors(cudaMemcpy(device_in,in,size_uc_pixel,cudaMemcpyHostToDevice));
	//
	/*cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<float>(); 
	cudaBindTexture(NULL,t_p_Mag,device_p_Mag,&channelDesc,size_c_pixel);
	cudaBindTexture(NULL,t_p_ANG,device_p_ANG,&channelDesc,size_c_pixel);*/
    checkCudaErrors(cudaMemset(device_out,0,size_c_cell));
    checkCudaErrors(cudaMemset(device_smooth_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_block_out,0,size_c_block));
    checkCudaErrors(cudaMemset(device_out_norm,0,size_c_block));

   
    int h_windowx=Imagewidth/128;
    int h_windowy=ImageHeight/128;
    //dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
	dim3 blocks(4,4);
	dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量
	//countCell<<<blocks,threads>>>(device_in, device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask);

	//dim3 reduce_block(16,1);
	//dim3 reduce_thread(128,10);
	//reduce<<<reduce_block,reduce_block>>>(device_out,device_smooth_out);
	//checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaMemcpy(out,device_out,1280*sizeof(float),cudaMemcpyDeviceToHost));
	dim3 block_right(32);
	dim3 thread_right(4,128);

    dim3 block_smooth(18,7);//一个cell分18个角度方向,一个方向7个cell，
    dim3 threads_smooth(10);//每个cell 10 个bin

	
    dim3 block_b(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
    dim3 thread_b(3,10);//3=一个block包含3个cell,10=一个cell10个bin

    dim3 block_norm(72);//blob的数量 18*4=72
    dim3 thread_norm(30);//block特征向量长度（m_nBIN）
    
 //  /* for(int i=0;i<h_windowy;i++)
 //       for(int j=0;j<h_windowx;j++)
	//	{ */      //smoothcell<<<block_smooth,threads_smooth>>>(device_out+(i*h_windowy+j)*1260,device_smooth_out);
	

	countCell<<<blocks,threads>>>( device_out, device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask,d_histo_mask);
	smoothcell<<<block_smooth,threads_smooth>>>(device_out,device_smooth_out);
	countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
	normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);
	cudaDeviceSynchronize();
	//checkCudaErrors(cudaMemcpy(t_FeatureMat->data,device_out_norm,sizeof(device_out_norm),cudaMemcpyDeviceToHost));
		 
			/* t_nRes=t_pSVM->predict(t_FeatureMat); 
			 cudaDeviceSynchronize();
			 printf(" %f ",t_nRes);*/
	//for(int j=12;j<Imagewidth-stride;j+=12){
	//			/*right(device_out,device_out,0,j,*/
	//		 //right<<<block_right,thread_right>>>(device_out,device_out,j,0,device_c_ANG,device_c_Mag,device_p_ANG,device_p_Mag,ImageHeight,Imagewidth,d_mask,d_histo_mask);
	//		 right<<<block_right,thread_right>>>(device_out,j,0,device_c_ANG,device_c_Mag,device_p_ANG,device_p_Mag,ImageHeight,Imagewidth,d_mask,d_histo_mask);
	//		 
			
		//
		/*	
				checkCudaErrors(cudaMemcpy(out+j*2160,device_out_norm,size_c_block,cudaMemcpyDeviceToHost));*/
		 //}
	//	for(int i=0;i<ImageHeight-stride;i++)
	//		{for(int j=0;j<Imagewidth-stride;j++)
	//		{
	///*countCell<<<blocks,threads>>>( device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask,d_histo_mask);
	//	*/	
	//			if(i==0)
	//				
	//			if(!i%2&&
	//				smoothcell<<<block_smooth,threads_smooth>>>(device_out,device_smooth_out);
	//			countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
	//			normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);
	//			cudaDeviceSynchronize();

	//	}
 //checkCudaErrors(cudaMemcpy(out,device_out_norm,size_c_block,cudaMemcpyDeviceToHost));
	//		}
   /* cudaFreeArray(cuArray_Mag);
	   cudaFreeArray(cuArray_ANG);*/
    cudaFree(device_c_ANG);
    cudaFree(device_c_Mag);
    /*cudaFree(device_d_ANG);
    cudaFree(device_d_Mag);*/
    cudaFree(device_p_ANG);
    cudaFree(device_p_Mag);
   
    cudaFree(device_out);
    cudaFree(device_smooth_out);
    cudaFree(device_block_out);
    cudaFree(device_out_norm);
 //cudaFree(device_in);


    cudaDeviceReset();
    
    
}