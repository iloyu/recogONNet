#include "countFeatures.cuh"
#define L2HYS_EPSILON 		0.01f
#define L2HYS_EPSILONHYS	1.0f
#define L2HYS_CLIP			0.2f
#define data_h2y            30
    //long h_windowx=Imagewidth/Windowx;
    //long h_windowy=ImageHeight/Windowy;
    //dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
    //dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量
__global__ void add(float *in,float *in2)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	in[tid]+=in2[tid];
}

__global__ void countCell(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int ImageHeight,int Imagewidth,int *mask)
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
	 t_fm_nbin[tidy][tidx]=p_ANG[(yy)*Imagewidth+xx]-c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
		 __syncthreads();
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		atomicAdd(& (histo[d_ANG[(yy)*m_nImage+xx]*70+d_Mag[(yy)*m_nImage+xx]*10+t_nm_nbin[tidy][tidx]]),
			p_Mag[(yy)*Imagewidth+xx]*mask[xx+(yy)*m_nImage]);
		
		__syncthreads();
   //out[d_ANG[(yy)*m_nImage+xx]*70+d_Mag[(yy)*m_nImage+xx]*10+t_nm_nbin[tidy][tidx]]=histo[d_ANG[(yy)*m_nImage+xx]*70+d_Mag[(yy)*m_nImage+xx]*10+t_nm_nbin[tidy][tidx]];
		atomicAdd(&out[tidy*32+tidx],histo[tidy*32+tidx]);
		//out[tidy*32+tidx+id*1280]=histo[tidy*32+tidx];
		if(tidy%4==0)
				atomicAdd(&out[1024+(tidy/4)*32+tidx],histo[1024+(tidy/4)*32+tidx]);
		__syncthreads();
		
		

}
	
//dim3 block(18,7);//一个cell分18个角度方向,一个方向7个cell，
	//dim3 threads(10);//每个cell 10 个bin
__global__ void reduce(float *in,float *out){
	//int id=blockIdx.y*gridDim.x+blockIdx.x;
	int tidx=threadIdx.x;
    int tidy=threadIdx.y;
	int yy=blockIdx.y*blockDim.y+tidy;
	int xx=blockIdx.x*blockDim.x+tidx;
	if(blockIdx.x<8) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*8];
	__syncthreads();
	if(blockIdx.x<4) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*4];
	__syncthreads();
	if(blockIdx.x<2) in[tidy*blockDim.x+tidx]+=in[tidy*blockDim.x+tidx+1280*2];
	__syncthreads();

	if(blockIdx.x==0)out[tidy*blockDim.x+tidx]=in[tidy*blockDim.x+tidx]+in[tidy*blockDim.x+tidx+1280];
	
//	for(int stride=1280*8;stride>1280;stride/=2)
//	{ if(idx_all<stride)
//	in[idx_all]+=in[idx_all+stride];
//		__syncthreads();
//}

}
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

__global__ void smooth(float *in,float *out)
{
	int k,j,i;
	int m_nBIN=10;
	float *m_pCellFeatures=in;
	int t_nLineWidth=70;
	float t_pTemp[10];
	for ( k = 0; k < 18; ++k )//18
	{
		for ( j = 0; j < 7; ++j )//7
		{
			for ( i = 0; i< 10; ++i )//10
			{
				int t_nLeft;
				int t_nRight;
				t_nLeft = ( i - 1 + m_nBIN ) % m_nBIN;
				t_nRight = ( i + 1 ) % m_nBIN;

				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.8f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nLeft] * 0.1f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nRight] * 0.1f;
			}

			for ( i = 0; i < m_nBIN; ++i )
			{
				out[k * t_nLineWidth + j * m_nBIN + i] = t_pTemp[i];
			}
		}
	}
	
}

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
    
__global__ void normalizeL2Hys(float *in,float *out)
{
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    // Sum the vector
    float sum = 0;
    
    float *t_ftemp=in+bid*30;
    float *t_foutemp=out+bid*30;
    sum+=t_ftemp[tid]*t_ftemp[tid];
    __syncthreads();
    // Compute the normalization term
    float norm = 1.0f/(rsqrt(sum) + L2HYS_EPSILONHYS * 30);
    t_foutemp[tid]=t_ftemp[tid]*norm;
    __syncthreads();


}
 extern "C" void countFeaturesfloat(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask)
{

    int *device_d_ANG,*device_d_Mag,*d_mask;
    float *device_c_ANG, *device_c_Mag,*device_p_Mag,*device_p_ANG,*device_out,*device_smooth_out,*device_block_out,*device_out_norm;
    uchar *device_in;

    long size_d_window=sizeof(int)*m_nImage*m_nImage;
    long size_c_window=sizeof(float)*m_nImage*m_nImage;
    long size_c_pixel=sizeof(float)*ImageHeight*Imagewidth;
    long size_uc_pixel=sizeof(uchar)*ImageHeight*Imagewidth;
    long size_c_cell=sizeof(float)*1280*(ImageHeight/m_nImage)*(Imagewidth/m_nImage);
    long size_s_cell=sizeof(float)*1280;
    long size_c_block=sizeof(float)*2160;

    checkCudaErrors(cudaMalloc((void **)&device_c_ANG,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_c_Mag,size_c_window));
    checkCudaErrors(cudaMalloc((void **)&device_d_ANG,size_d_window));
    checkCudaErrors(cudaMalloc((void **)&device_d_Mag,size_d_window));
    checkCudaErrors(cudaMalloc((void **)&device_p_ANG,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_p_Mag,size_c_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_in,size_uc_pixel));
    checkCudaErrors(cudaMalloc((void **)&device_out,size_c_cell));
    checkCudaErrors(cudaMalloc((void **)&device_smooth_out,size_s_cell));
    checkCudaErrors(cudaMalloc((void **)&device_block_out,size_c_block));
    checkCudaErrors(cudaMalloc((void **)&device_out_norm,size_c_block));
	checkCudaErrors(cudaMalloc((void **)&d_mask,size_d_window));

    checkCudaErrors(cudaMemcpy(device_c_ANG,c_ANG,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_c_Mag,c_Mag,size_c_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_d_Mag,d_Mag,size_d_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_d_ANG,d_ANG,size_d_window,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_Mag,p_Mag,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_p_ANG,p_ANG,size_c_pixel,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_in,in,size_uc_pixel,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mask,mask,size_d_window,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(device_out,0,size_c_cell));
    checkCudaErrors(cudaMemset(device_smooth_out,0,size_s_cell));
    checkCudaErrors(cudaMemset(device_block_out,0,size_c_block));
    checkCudaErrors(cudaMemset(device_out_norm,0,size_c_block));

   
   /* int h_windowx=Imagewidth/128;
    int h_windowy=ImageHeight/128;*/
    //dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
	dim3 blocks(4,4);
	dim3 threads(Windowx,Windowy);//每一个线程块计算一个cell的特征量
	countCell<<<blocks,threads>>>(device_in, device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, ImageHeight,Imagewidth,d_mask);
	//
	////getLastCudaError();
	//dim3 reduce_block(16,1);
	//dim3 reduce_thread(128,10);
	//reduce<<<reduce_block,reduce_block>>>(device_out,device_smooth_out);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(out,device_out,1280*sizeof(float),cudaMemcpyDeviceToHost));
	
    dim3 block_smooth(18,7);//一个cell分18个角度方向,一个方向7个cell，
    dim3 threads_smooth(10);//每个cell 10 个bin

	
    dim3 block_b(18,4);//18=m_nANG一个窗口分18个方向，4=一个角度方向4个block
    dim3 thread_b(3,10);//3=一个block包含3个cell,10=一个cell10个bin

    dim3 block_norm(72);//blob的数量 18*4=72
    dim3 thread_norm(30);//block特征向量长度（m_nBIN）
    
 //   for(int i=0;i<h_windowy;i++)
 //       for(int j=0;j<h_windowx;j++)
	//	{       //smoothcell<<<block_smooth,threads_smooth>>>(device_out+(i*h_windowy+j)*1260,device_smooth_out);
 //            //countCell<<<blocks,threads>>>(device_in, device_out, device_d_ANG,device_d_Mag,device_c_ANG, device_c_Mag, device_p_ANG, device_p_Mag, Imagewidth,ImageHeight);    
	//		smooth<<<1,1>>>(device_out+(i*h_windowx+j)*1280,device_smooth_out);
	//		countblock<<<block_b,thread_b>>>(device_smooth_out,device_block_out);
 //               normalizeL2Hys<<<block_norm,thread_norm>>>(device_block_out,device_out_norm);
 //cudaDeviceSynchronize();
	//			checkCudaErrors(cudaMemcpy(out+(i*h_windowx+j)*2160,device_out_norm,size_c_block,cudaMemcpyDeviceToHost));
 //              
 //   }

    cudaFree(device_c_ANG);
    cudaFree(device_c_Mag);
    cudaFree(device_d_ANG);
    cudaFree(device_d_Mag);
    cudaFree(device_p_ANG);
    cudaFree(device_p_Mag);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_smooth_out);
    cudaFree(device_block_out);
    cudaFree(device_out_norm);



    cudaDeviceReset();
    
    
}