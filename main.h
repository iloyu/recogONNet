#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
using namespace cv;
#define Pi 3.1415926535897f
#define windowx 128
#define feature 1260
extern "C" void countFeaturesfloat(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask);
void CountCell( float * m_pCellFeatures,int width, int height, int t_nWidth, int t_nHeight, int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int *mask)

{

	//int t_nEndX;
	//int t_nEndY;
	//t_nEndX = t_nX + t_nWidth - 1;
	//t_nEndY = t_nY + t_nHeight - 1;

	int t_nCellFeatureSize;
	t_nCellFeatureSize = 18*7*10;//m_nANG * m_nCellNumb * m_nBIN;
	int i, j;
	for ( i = 0; i < t_nCellFeatureSize; ++i )
	{
		m_pCellFeatures[i] = 0;
	}
	//生成cellfeature

	int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = 70;
	for ( j = 0; j < t_nHeight; ++j )
	{
		for ( i = 0; i < t_nWidth; ++i )
		{
			//判断是否超出半径
			/*if ( c_Mag[j*128+i] > 64 )
			{
				continue;
			}*/
			//计算m_nBIN

			float t_fm_nANGel;
			t_fm_nANGel = p_ANG[j*width+i] - c_ANG[j*128+i];
			while ( t_fm_nANGel < 0 )

			{

				t_fm_nANGel += (float)Pi;

			}



			int t_nm_nBIN =  (int)( t_fm_nANGel * 10 / Pi);
			m_pCellFeatures[t_nLineWidth * d_ANG[ j * 128 + i ] + d_Mag[j* 128 + i ] * 10 + t_nm_nBIN ] += p_Mag[i+j*width]*mask[i+j*128];

		}

	}

}//CountCell
int main()
{
	Mat src=imread("C:\\Users\\Cyj\\Desktop\\test.jpg",CV_LOAD_IMAGE_UNCHANGED);
	cvtColor(src,src,CV_BGR2GRAY);
	float *dst,*m_fNormalMat,*m_fMagMat,*m_ANGImage,*m_MagImage,*cpu_out;
	int *m_nANG,*m_nMag,*mask;
	int width=src.cols;
	int height=src.rows;
	long sizem_p=sizeof(float)*width*height;
	long sizem_n=sizeof(int)*128*128;
	long sizem_f=sizeof(float)*128*128;
	long size_cpu=sizeof(float)*1260;
	long size_mask=sizeof(int)*128*128;
	m_nANG=(int *)malloc(sizem_n);
	m_nMag=(int *)malloc(sizem_n);
	m_fMagMat=(float *)malloc(sizem_f);
	m_fNormalMat=(float*)malloc(sizem_f);
	m_MagImage=(float*)malloc(sizem_p);
	m_ANGImage=(float*)malloc(sizem_p);
	cpu_out=(float *)malloc(size_cpu);
	mask=(int*)malloc(size_mask);
	memset(cpu_out,0,size_cpu);
	memset(mask,0,size_mask);
	int h_windowx=src.cols/128;
	int h_windowy=src.rows/128;
	dst=(float*)malloc(sizeof(float)*h_windowx*h_windowy*feature);
	//dst=(float*)malloc(sizeof(float)*1260);
	FILE *fp;
	//=fopen("G:\\Mag.txt","r");//打开文件以便从中读取数据
	
	float *p_Mag,*p_ANG;
 //for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
 //   for(int j=0;j<width;j++)
	// fscanf(fp," %f ",&p_Mag[i*width+j]);
 //}
 //fclose(fp);
 //fp=fopen("G://ANG.txt","r");
 //for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
 //   for(int j=0;j<width;j++)
	// fscanf(fp," %f ",&p_ANG[i*width+j]);
 //}
 //fclose(fp);
 fp=fopen("G://m_nANg.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		fscanf(fp," %d ",&m_nANG[i*windowx+j]);
 }
 fclose(fp);
  fp=fopen("G://m_nMag.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		fscanf(fp," %d ",&m_nMag[i*windowx+j]);
 }
 fclose(fp);
 float max=0;
  fp=fopen("G://m_nfNormalMat.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		{fscanf(fp," %f ",&m_fNormalMat[i*windowx+j]);
	max=MAX(m_fNormalMat[i*windowx+j],max);
	}
	}
 

 fclose(fp);
 //printf("max:%f",max);
  fp=fopen("G://m_nfMagMat.txt","r");long count=0;
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		{fscanf(fp," %f ",&m_fMagMat[i*windowx+j]);
	    if(m_fMagMat[i*windowx+j]<64)
			mask[i*windowx+j]=1;
	}

 }
 //printf(" %d ",count);
 fclose(fp);
 fp=fopen("G://m_ANGLEPixel.txt","r");
 for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
		{fscanf(fp," %f ",&m_ANGImage [i*width+j]);
	
	}
 }
 fclose(fp);
 
 fp=fopen("G://m_MagPixel.txt","r");
	for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
				{fscanf(fp," %f ",&m_MagImage[i*width+j]);
	if(m_MagImage[i*width+j]<0)
		m_MagImage[i*width+j]=0;
	}
 }
 fclose(fp);
 fp=fopen("G://m_MagPixel.txt","w");
 for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
		{fprintf(fp," %f ",m_MagImage[i*width+j]);
	//printf("%f\n",m_MagImage);
	}
 }
 fclose(fp);
 countFeaturesfloat(src.data,dst,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,width,height,mask);
	fp=fopen("G://out.txt","w");
	for (int i = 0; i < h_windowy; i++) {  //从fp指向的文件中读取10个整数到b数组
		for(int j=0;j<h_windowx;j++)
			for (int k=0;k<feature;k++)
				fprintf(fp," %f\n",dst[((i*h_windowx)+j)*feature+k]);
 }
 fclose(fp);
 CountCell(cpu_out,width,height,128,128,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask);
 fp=fopen("G://out_cpu.txt","w");
 for (int i = 0; i <feature; i++) {  //从fp指向的文件中读取10个整数到b数组
		
		fprintf(fp," %f\n",cpu_out[i]);
 }
 fclose(fp);
 fp=fopen("G://mask.txt","w");
 for (int i = 0; i <128*128; i++) {  //从fp指向的文件中读取10个整数到b数组
		
		fprintf(fp," %d\n",mask[i]);
 }
 fclose(fp);
 	free(m_nANG);
	free(m_nMag);
	free(m_fMagMat);
	free(m_fNormalMat);
	free(m_MagImage);//=(float*)malloc(sizem_p);
	free(m_ANGImage);//=(float*)malloc(sizem_p);
	free(cpu_out);//=(float *)malloc(size_cpu);
	free(mask);//=(float*)malloc(size_mask);
 return 0;
}