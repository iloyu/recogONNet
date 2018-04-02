/*
dim3 thread(4,128);一个移动步长为12，需要移动两次
dim3 block(32);
right move
*/
__global__ void right(float *in,float *out,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_hist_mask){
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=threadIdx.y;
	
	__shared__  float t_fm_nbin[4][128];
	
	__shared__  int  t_nm_nbin[4][128];
	if(blockIdx.x<4)
	{  t_fm_nbin[tidy][tidx]=device_p_ANG[yy*Imagewidth+xx]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		atomicSub(& (in[d_hist_mask[m_nImage*yy+x]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);
	}
			__syncthreads();
			if(blockIdx.x>28)
			{   t_fm_nbin[tidy][tidx]=device_p_ANG[yy*Imagewidth+xx]-device_c_ANG[(yy)*m_nImage+xx];
				if( t_fm_nbin[tidy][tidx]<0)
					atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
			if( t_fm_nbin[tidy][tidx]<0)
					atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
			t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
			atomicAdd(& (in[d_hist_mask[m_nImage*yy+x]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);
			
	}	
			
}
__global__ void left(float *in,float *out,float *device_c_ANG,float *device_c_Mag,float *device_p_ANG,float *device_p_Mag,int ImageHeight,int Imagewidth,int *d_mask,int *d_hist_mask){
	int xx=blockIdx.x*blockDim.x+threadIdx.x;
	int yy=threadIdx.y;
	
	__shared__  float t_fm_nbin[4][128];
	
	__shared__  int  t_nm_nbin[4][128];
	if(blockIdx.x<4)
	{  t_fm_nbin[tidy][tidx]=device_p_ANG[yy*Imagewidth+xx]-device_c_ANG[(yy)*m_nImage+xx];
	    if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 if( t_fm_nbin[tidy][tidx]<0)
         atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
		 t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
		atomicAdd(& (in[d_hist_mask[m_nImage*yy+x]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);
	}
			__syncthreads();
			if(blockIdx.x>28)
			{if( t_fm_nbin[tidy][tidx]<0)
					atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
			if( t_fm_nbin[tidy][tidx]<0)
					atomicAdd(&t_fm_nbin[tidy][tidx],Pi); 
		  
			t_nm_nbin[tidy][tidx]=(int)(t_fm_nbin[tidy][tidx]*10/Pi);
	
		 //out[tidy*32+tidx]=t_nm_nbin[tidy][tidx];
			atomicAdd(& (in[d_hist_mask[m_nImage*yy+x]+t_nm_nbin[tidy][tidx]]),
			device_p_Mag[(yy)*Imagewidth+ xx]*d_mask[xx+(yy)*m_nImage]);
			
	}	
			
}
}