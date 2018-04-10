#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include "opencv2/ml/ml.hpp"
#include <windows.h>
#include "time.h"
#include "ROIDef.h"

#include "SomeFunction.h"
using namespace cv;
#define Pi 3.1415926535897f
#define windowx 128
#define my_feature 2160
#define precision 0.1

float *dst,*m_fNormalMat,*m_fMagMat,*m_ANGImage,*m_MagImage,*cpu_out,*cpu_smooth_out,
		*cpu_block_out,*my_feat,*re_feat;
	int *m_nANG,*m_nMag,*mask,*hist_mask;
	float t_nRes;

 extern "C" void countFeaturesfloat(float *out,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,
	 int *mask,int *histo_mask,float SVM_bias,float *svm_weight,int svm_count,int Imagewidth,int ImageHeight,
	 float t_fAugRate,vector <CvRect> &t_vTarget);
void CountCell( float * m_pCellFeatures,int t_nY,int t_nX,int width, int height, int t_nWidth, 
int t_nHeight, int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int *mask)
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
void testTime()
{
	_LARGE_INTEGER time_start;  //开始时间  
	_LARGE_INTEGER time_over;   //结束时间  
	double dqFreq;      //计时器频率  
	LARGE_INTEGER f;    //计时器频率  
	QueryPerformanceFrequency(&f);  
	dqFreq=(double)f.QuadPart;  
//	QueryPerformanceCounter(&time_start);
//CountCell(cpu_out,0,0,width,height,128,128,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask);
//QueryPerformanceCounter(&time_over);    //计时结束  
//	float time_elapsed=1000000*(time_over.QuadPart-time_start.QuadPart)/dqFreq;  
//	//乘以1000000把单位由秒化为微秒，精度为1000 000/（cpu主频）微秒  
//
//SmoothCell(cpu_out);
// Countm_nBIN(4,3,10,7,18,cpu_block_out,cpu_out);
// //norm2(3,10,18,4,cpu_block_out);


 
}
void testcell(int width,int height)
{
	
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
	my_feat=(float*)malloc(size_block);
	re_feat=(float*)malloc(size_block);

	memset(cpu_out,0,size_cpu);
	memset(mask,0,size_mask);
	
	dst=(float*)malloc(sizeof(float)*my_feature);
	FILE *fp;
	
	float *p_Mag,*p_ANG;

	fp=fopen("G://my_feat.txt","r");

    for(int j=0;j<2160;j++)
		{fscanf(fp," %f ",&my_feat[j]);
	/*hist_mask[i*windowx+j]=m_nANG[i*windowx+j]*70+m_nMag[i*windowx+j]*10;*/
	}
	fclose(fp);
		fp=fopen("G://re_feat.txt","r");

    for(int j=0;j<2160;j++)
		{fscanf(fp," %f ",&re_feat[j]);
	/*hist_mask[i*windowx+j]=m_nANG[i*windowx+j]*70+m_nMag[i*windowx+j]*10;*/
	}
	fclose(fp);
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

 //float max=0;
  fp=fopen("G://m_nfNormalMat.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		{fscanf(fp," %f ",&m_fNormalMat[i*windowx+j]);
	//max=MAX(m_fNormalMat[i*windowx+j],max);
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
 
 fp=fopen("G://t_Mag.txt","r");
	for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
				{fscanf(fp," %f",&m_MagImage[i*width+j]);
	printf("%f ",m_MagImage[i*width+j]);
	if(m_MagImage[i*width+j]<0)
		m_MagImage[i*width+j]=0;
	}
 }
 fclose(fp);


 CountCell(cpu_out,height-256,0,width,height,128,128,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask);
 //SmoothCell(cpu_out);
 Countm_nBIN(4,3,10,7,18,cpu_block_out,cpu_out);
 norm2(3,10,18,4,cpu_block_out);
 fp=fopen("G://out_cpu.txt","w");
 for (int i = 0; i <1260; i++) {  //从fp指向的文件中读取10个整数到b数组
			fprintf(fp," %f ",cpu_out[i]);
		
		
		}
 //for (int i = 0; i <18; i++) {  //从fp指向的文件中读取10个整数到b数组
	//	{for(int j=0;j<120;j++)
	//		fprintf(fp," %f ",cpu_block_out[i*120+j]);
	//	}
	//	fprintf(fp,"\n");
	//	}
		
 fclose(fp);
 /*printf("countcell: %lf us",(double)time_elapsed);
 //getchar();*/
 //for (int i = 0; i <18; i++) {  //从fp指向的文件中读取10个整数到b数组
	//	for(int j=0;j<120;j++)
	//	{ 
	//		//if(cpu_block_out[i*120+j]-dst[i*120+j]>precision||cpu_block_out[i*120+j]-dst[i*120+j]<-precision)
	//			//printf("%d,%d:%f,%f ",i,j,cpu_block_out[i*120+j],dst[i*120+j]);
	//	}
	//	printf("\n");
	//	}
	/*for(int j=0;j<2160;j++)
		{ 
			if(re_feat[j]-my_feat[j]>precision||re_feat[j]-my_feat[j]<-precision)
				printf("%d:%f,%f ",j,my_feat[j],re_feat[j]);
}*/
}
void release()
{
		
 	free(m_nANG);
	free(m_nMag);
	free(m_fMagMat);
	free(m_fNormalMat);
	free(m_MagImage);//=(float*)malloc(sizem_p);
	free(m_ANGImage);//=(float*)malloc(sizem_p);
	free(cpu_out);//=(float *)malloc(size_cpu);
	free(mask);//=(float*)malloc(size_mask);
	free(hist_mask);
}
class TargetArea
{
public:
	TargetArea()
	{
		m_nDupeNumber = 0;
		m_nCenterX = 0;
		m_nCenterY = 0;
		m_nWidth = 0;
		m_nHeight = 0;
	}

	~TargetArea(){};

public:
	int m_nDupeNumber;
	int m_nCenterX;
	int m_nCenterY;

	int m_nWidth;
	int m_nHeight;
};

int RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime )	
{
	int t_nRectNum;	
	t_nRectNum = (int)t_vTarget.size();
	clock_t start,stop;
	_LARGE_INTEGER time_start,s1,s2,s3;  //开始时间  
	_LARGE_INTEGER time_over,e1,e2,e3;   //结束时间  
	double dqFreq;      //计时器频率  
	LARGE_INTEGER f;    //计时器频率  
	QueryPerformanceFrequency(&f);  
	dqFreq=(double)f.QuadPart;  
	QueryPerformanceCounter(&time_start);
	//聚类区域
	vector <TargetArea> t_vAreaSeq;
	int i, j;
	start=clock();
	for ( i = 0; i < t_nRectNum; ++i )
	{
		int t_nCenterX;
		int t_nCenterY;
		t_nCenterX = t_vTarget[i].x;	// + t_vTarget[i].width / 2;
		t_nCenterY = t_vTarget[i].y;	// + t_vTarget[i].height / 2;
		bool t_bFinded;		//是否找到匹配目标
		t_bFinded = false;
		for( j = 0; j < (int)t_vAreaSeq.size(); ++j )
		{
			if ( abs( t_nCenterX - t_vAreaSeq[j].m_nCenterX ) < t_vTarget[i].width / 3
				&& abs( t_nCenterY - t_vAreaSeq[j].m_nCenterY ) < t_vTarget[i].height / 3
				&& t_vAreaSeq[j].m_nWidth == t_vTarget[i].width )
			{
				t_vAreaSeq[j].m_nCenterX = ( t_vAreaSeq[j].m_nCenterX * t_vAreaSeq[j].m_nDupeNumber + t_nCenterX ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nCenterY = ( t_vAreaSeq[j].m_nCenterY * t_vAreaSeq[j].m_nDupeNumber + t_nCenterY ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nWidth = ( t_vAreaSeq[j].m_nWidth * t_vAreaSeq[j].m_nDupeNumber + t_vTarget[i].width ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nHeight = ( t_vAreaSeq[j].m_nHeight * t_vAreaSeq[j].m_nDupeNumber + t_vTarget[i].height ) / ( t_vAreaSeq[j].m_nDupeNumber + 1 );
				t_vAreaSeq[j].m_nDupeNumber ++;
				t_bFinded = true;
				break;
			}
		}
		if ( !t_bFinded )
		{
			TargetArea t_TarAdd;
			t_TarAdd.m_nCenterX = t_nCenterX;
			t_TarAdd.m_nCenterY = t_nCenterY;
			t_TarAdd.m_nWidth = t_vTarget[i].width;
			t_TarAdd.m_nHeight = t_vTarget[i].height;
			t_TarAdd.m_nDupeNumber = 1;
			t_vAreaSeq.push_back( t_TarAdd );
		}
	}
	//删除孤立区域
	int t_nTarNUmber;
	t_nTarNUmber = (int)t_vAreaSeq.size();
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber >= 2 )
		{
			t_vAreaSeq[i].m_nDupeNumber += 1;
		}
	}
	//合并区域
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber <= 0 )
		{
			continue;
		}
		for ( j = i + 1; j < (int)t_vAreaSeq.size(); ++j )
		{
			if ( t_vAreaSeq[j].m_nDupeNumber <= 0 )
			{
				continue;
			}
			if ( abs( t_vAreaSeq[i].m_nCenterX - t_vAreaSeq[j].m_nCenterX ) < ( t_vAreaSeq[i].m_nWidth + t_vAreaSeq[j].m_nWidth ) / 3
				&& abs( t_vAreaSeq[i].m_nCenterY - t_vAreaSeq[j].m_nCenterY ) < ( t_vAreaSeq[i].m_nWidth + t_vAreaSeq[j].m_nWidth ) / 3 )
			{
				t_vAreaSeq[i].m_nCenterX = ( t_vAreaSeq[i].m_nCenterX * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nCenterX * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nCenterY = ( t_vAreaSeq[i].m_nCenterY * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nCenterY * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nWidth = ( t_vAreaSeq[i].m_nWidth * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nWidth * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nHeight = ( t_vAreaSeq[i].m_nHeight * t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nHeight * t_vAreaSeq[j].m_nDupeNumber ) / ( t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber );
				t_vAreaSeq[i].m_nDupeNumber = t_vAreaSeq[i].m_nDupeNumber + t_vAreaSeq[j].m_nDupeNumber;
				t_vAreaSeq[j].m_nDupeNumber = -1;
				t_nTarNUmber--;
			}
		}
	}
	//删除较低概率区域
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber < t_nMatchTime && t_vAreaSeq[i].m_nDupeNumber > 0 )
		{
			t_vAreaSeq[i].m_nDupeNumber = -1;
			t_nTarNUmber--;
		}
	}
	//保留数据
	if ( t_nTarNUmber <= 0 )		//如果没有目标，直接返回NULL
	{
		return 0;
	}
	t_pRect = new iRect[t_nTarNUmber];
	j = 0;
	for ( i = 0; i < (int)t_vAreaSeq.size(); ++i )
	{
		if ( t_vAreaSeq[i].m_nDupeNumber > 0 )
		{
			t_pRect[j].x = t_vAreaSeq[i].m_nCenterX - t_vAreaSeq[i].m_nWidth;
			t_pRect[j].y = t_vAreaSeq[i].m_nCenterY - t_vAreaSeq[i].m_nHeight;
			t_pRect[j].m_nWidth = t_vAreaSeq[i].m_nWidth * 2 - 2;
			t_pRect[j].m_nHeight = t_vAreaSeq[i].m_nHeight * 2 - 2;
			j++;
		}
	}
		
	return t_nTarNUmber;
}
int main()
{
	Mat pre;
	FILE *fp;
	string c_name="C:\\Users\\Cyj\\Desktop\\test.jpg";
	Mat src=imread("C:\\Users\\Cyj\\Desktop\\test.jpg",CV_LOAD_IMAGE_UNCHANGED);
	cvtColor(src,src,CV_BGR2GRAY);
	medianBlur( src, src, 7 );
	GaussianBlur( src, src, cvSize( 3, 3 ), 1 );
	int i, j;
	int width=src.cols;
	int height=src.rows;
 
	CvSVM  t_pSVM ;string t_sClassFilePath="C:\\Users\\Cyj\\Desktop\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
		/*m_pClassifier = (void *)t_pSVM;*/

	int svm_count=t_pSVM.get_var_count();
	float svm_bias=-2.7828561096231148e+000;
		float *svm_weights;
		 float t_fAugRate=1;
		svm_weights=(float *)malloc(sizeof(float)*svm_count);
 //memset(svm_weights,0,sizeof(float)*2160);
 i=0;
 float res=0;
 testcell(width,height);
 while(i<svm_count)
 {
	 svm_weights[i]=t_pSVM.get_support_vector(0)[i];
     i++;
 } 
 for(i=0;i<2160;i++)
	 res+=svm_weights[i]*cpu_block_out[i];
 //printf("res: %f",res);
 vector <CvRect> t_vTarget;
 countFeaturesfloat(dst,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask,svm_bias,
	 svm_weights,svm_count,width,height,t_fAugRate,t_vTarget);
 fp=fopen("G://out.txt","w");
for (int k=0;k<2160;k++)
	fprintf(fp," %f ",dst[k]);
 fclose(fp);
 IplImage *t_Image=NULL;
 iRect *t_Rect=NULL;
 ListImage tmpListImage;
 tmpListImage.LoadImageFromFile(c_name.c_str());
		cvtList2Ipl(&tmpListImage,t_Image);
		int num=RefineTargetSeq(t_vTarget,t_Rect,5);
		printf("%d",num);
		for (j=0;j<num;j++)
		{
			cvRectangle(t_Image,cvPoint(t_Rect[j].x,t_Rect[j].y),cvPoint(t_Rect[j].x+t_Rect[j].m_nWidth,t_Rect[j].y+t_Rect[j].m_nHeight),cvScalar(0xff,0x00,0x00),2);
		 cvShowImage("IplImage",t_Image); 
		 waitKey();
		}

	//	if (t_Rect!=NULL)
	//{
	//	delete[] t_Rect;
	//	t_Rect=NULL;
	//}

	return 0;
}