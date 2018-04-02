#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include "opencv2/ml/ml.hpp"
using namespace cv;
#define Pi 3.1415926535897f
#define windowx 128
#define feature 2160
#define precision 0.25
extern "C" void countFeaturesfloat(float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight,int *mask,int *histo_mask);
void CountCell( float * m_pCellFeatures,int t_nY,int t_nX,int width, int height, int t_nWidth, int t_nHeight, int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int *mask)
{

	int t_nEndX;
	int t_nEndY;
	t_nEndX = t_nX + t_nWidth ;
	t_nEndY = t_nY + t_nHeight ;

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
	int k;
	for ( j = t_nY; j <t_nEndY; ++j )
	{
		for ( i = t_nX; i < t_nEndX; ++i )
		{
			int off_x=i-t_nX;
			int off_y=j-t_nY;
			float t_fm_nANGel;
			t_fm_nANGel = p_ANG[j*width+i] - c_ANG[off_y*128+off_x];
			while ( t_fm_nANGel < 0 )
			{
				t_fm_nANGel += (float)Pi;

			}
			int t_nm_nBIN =  (int)( t_fm_nANGel * 10 / Pi);
			//printf(" %d ",t_nm_nBIN);
			
			m_pCellFeatures[t_nLineWidth * d_ANG[ off_y* 128 + off_x ] + d_Mag[off_y* 128 + off_x ] * 10 + t_nm_nBIN  ] += p_Mag[i+j*width]*mask[off_x+off_y*128];
		}
		//printf("\n");

	}

}
void SmoothCell(float *m_pCellFeatures)
{int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = 70;
	int i, j, k;
	float * t_pTemp;		//临时保存
	t_pTemp = new float [10];
	for ( k = 0; k < 18; ++k )//18
	{
		for ( j = 0; j < 7; ++j )//7
		{
			for ( i = 0; i< 10; ++i )//10
			{
				int t_nLeft;
				int t_nRight;
				t_nLeft = ( i - 1 + 10 ) % 10;
				t_nRight = ( i + 1 ) % 10;
				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * 10+ i] * 0.8f 
					+ m_pCellFeatures[k * t_nLineWidth + j * 10 + t_nLeft] * 0.1f 
					+ m_pCellFeatures[k * t_nLineWidth + j * 10 + t_nRight] * 0.1f;
			}
			for ( i = 0; i < 10; ++i )
			{
				m_pCellFeatures[k * t_nLineWidth + j * 10 + i] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//SmoothCell
void Countm_nBIN( int m_nBlobNumb,int m_nCellPerBlob,int m_nBIN,int m_nCellNumb,int m_nANG,float *m_pfFeature,float *m_pCellFeatures)
{
	int t_nLineWidthBlob;		//每个方向的宽度
	t_nLineWidthBlob = m_nBlobNumb * m_nCellPerBlob * m_nBIN;//4*3*10
	int t_nBlobWidth;
	t_nBlobWidth = m_nCellPerBlob * m_nBIN;//3*10
	int t_nLineWidthCell;		//cell每个方向的宽度
	t_nLineWidthCell = m_nCellNumb * m_nBIN;//7*10
	int i, j, k;
	for ( k = 0; k < m_nANG; ++k )
	{
		for ( j = 0; j < m_nBlobNumb; ++j )
		{
			for ( i = 0; i< m_nCellPerBlob; ++i )
			{
				memcpy( &m_pfFeature[k * t_nLineWidthBlob + j * t_nBlobWidth + i * m_nBIN], 
					&m_pCellFeatures[k * t_nLineWidthCell + (i + j) * m_nBIN], 
					m_nBIN * sizeof( float ) );
			}
		}
	}
}//Countm_nBIN
void norm2(int m_nCellPerBlob,int m_nBIN,int m_nANG,int m_nBlobNumb,float *m_pfFeature )
{
	int i, j;
	int t_nDataSize;		//特征向量长度（m_nBIN）
	t_nDataSize = m_nCellPerBlob * m_nBIN;//30
	int t_nm_nBlobNumber;		//blob的数量
	t_nm_nBlobNumber = m_nANG * m_nBlobNumb;//18*4=72
	for ( j = 0; j < t_nm_nBlobNumber; ++j )
	{
		//统计L2-normal分母
		float * t_fPos;		//数据访问指针
		t_fPos = &m_pfFeature[j * t_nDataSize];
		float t_fAddUp;
		t_fAddUp = 0;
		for ( i = 0; i < t_nDataSize; ++i )
		{
			//if ( t_fAddUp < t_fPos[i] )	//可替换项，最大值归一化
			//{
			//	t_fAddUp = t_fPos[i];
			//}
			t_fAddUp += t_fPos[i] * t_fPos[i];
		}
		t_fAddUp = sqrt( t_fAddUp + 1.0f );
		//t_fAddUp += 0.1f;		//可替换项，最大值归一化
		for ( i = 0; i < t_nDataSize; ++i )
		{
			t_fPos[i] = t_fPos[i] / t_fAddUp;
		}
	}
}

int main()
{Mat pre;

	Mat src=imread("C:\\Users\\Cyj\\Desktop\\test.jpg",CV_LOAD_IMAGE_UNCHANGED);
	cvtColor(src,src,CV_BGR2GRAY);
	medianBlur( src, src, 7 );
	GaussianBlur( src, src, cvSize( 3, 3 ), 1 );
	int i, j;
	
	float *dst,*m_fNormalMat,*m_fMagMat,*m_ANGImage,*m_MagImage,*cpu_out,*cpu_smooth_out,*cpu_block_out;
	int *m_nANG,*m_nMag,*mask,*hist_mask;
	float t_nRes;//SVM分类后的概率
	/*CvMat *t_FeatureMat;*/
	CvSVM  t_pSVM ;string t_sClassFilePath="C:\\Users\\Cyj\\Desktop\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
		/*m_pClassifier = (void *)t_pSVM;*/
	

	int width=src.cols;
	int height=src.rows;
	long sizem_p=sizeof(float)*width*height;
	long sizem_n=sizeof(int)*128*128;
	long sizem_f=sizeof(float)*128*128;
	long size_cpu=sizeof(float)*1260;
	long size_mask=sizeof(int)*128*128;
	long size_block=sizeof(float)*2160;
	long size_hist_mask=sizeof(int)*128*128;

	m_nANG=(int *)malloc(sizem_n);
	m_nMag=(int *)malloc(sizem_n);
	m_fMagMat=(float *)malloc(sizem_f);
	m_fNormalMat=(float*)malloc(sizem_f);
	m_MagImage=(float*)malloc(sizem_p);
	m_ANGImage=(float*)malloc(sizem_p);
	cpu_out=(float *)malloc(size_cpu);
	cpu_smooth_out=(float*)malloc(size_cpu);
	cpu_block_out=(float *)malloc(size_block);
	hist_mask=(int*)malloc(size_hist_mask);
	mask=(int*)malloc(size_mask);

	memset(cpu_out,0,size_cpu);
	memset(mask,0,size_mask);
	int h_windowx=src.cols/12;
	int h_windowy=src.rows/12;
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
		{fscanf(fp," %d ",&m_nMag[i*windowx+j]);
	/*hist_mask[i*windowx+j]=m_nANG[i*windowx+j]*70+m_nMag[i*windowx+j]*10;*/
	}
 }
 fclose(fp);
  fp=fopen("G://m_nhisto.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
	{fscanf(fp," %d ",&hist_mask[i*windowx+j]);
	}
 }
 fclose(fp);
 // fp=fopen("G://m_nhisto.txt","w");
 //for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
 //   for(int j=0;j<windowx;j++)
	//{fprintf(fp," %d ",hist_mask[i*windowx+j]);
	//}
 //}
 //fclose(fp);
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

 countFeaturesfloat(dst,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,width,height,mask,hist_mask);
fp=fopen("G://out.txt","w");
for (int i = 0; i <18; i++) {  //从fp指向的文件中读取10个整数到b数组
		{for(int j=0;j<120;j++)
		fprintf(fp," %f ",dst[i*120+j]);
		}
		fprintf(fp,"\n");
}


 fclose(fp);	

 for(int i=0;i<height-128;i+=12)
	 for(int j=0;j<width-128;j+=12)
 {CountCell(cpu_out,i,j,width,height,128,128,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask);
 SmoothCell(cpu_out);
 Countm_nBIN(4,3,10,7,18,cpu_block_out,cpu_out);
 norm2(3,10,18,4,cpu_block_out);
  CvMat* t_FeatureMat=&cvMat(1,2160,CV_32FC1,cpu_block_out) ;
  //IplImage* pImg =cvCreateImage(cvGetSize(t_FeatureMat), IPL_DEPTH_8U, 0);  
  //cvGetImage(t_FeatureMat,pImg);//从mat到img  
  //cvShowImage("hello",pImg);
  //cvWaitKey();
 t_nRes=t_pSVM.predict(t_FeatureMat);
		
			 printf(" %f ",t_nRes);
 }
 fp=fopen("G://out_cpu.txt","w");
 for (int i = 0; i <18; i++) {  //从fp指向的文件中读取10个整数到b数组
		{for(int j=0;j<120;j++)
		fprintf(fp," %f ",cpu_block_out[i*120+j]);
		}
		fprintf(fp,"\n");
		}
		
 fclose(fp);

 //for (int i = 0; i <18; i++) {  //从fp指向的文件中读取10个整数到b数组
	//	for(int j=0;j<120;j++)
	//	{ 
	//		if(cpu_block_out[i*120+j]-dst[i*120+j]>precision||cpu_block_out[i*120+j]-dst[i*120+j]<-precision)
	//			printf("%d,%d:%f,%f ",i,j,cpu_block_out[i*120+j],dst[i*120+j]);
	//	}
	//	printf("\n");
	//	}
 ////fp=fopen("G://mask.txt","w");
 //for (int i = 0; i <128*128; i++) {  //从fp指向的文件中读取10个整数到b数组
	//	
	//	fprintf(fp," %d\n",mask[i]);
 //}
 fclose(fp);
 	free(m_nANG);
	free(m_nMag);
	free(m_fMagMat);
	free(m_fNormalMat);
	free(m_MagImage);//=(float*)malloc(sizem_p);
	free(m_ANGImage);//=(float*)malloc(sizem_p);
	free(cpu_out);//=(float *)malloc(size_cpu);
	free(mask);//=(float*)malloc(size_mask);
	free(hist_mask);
	return 0;
}