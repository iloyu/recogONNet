#include "sourcev2.h"
using namespace cv;


float *dst,*m_fNormalMat,*m_fMagMat,*m_ANGImage,*m_MagImage,*cpu_out,*cpu_smooth_out,
*cpu_block_out,*my_feat,*re_feat;

int *m_nANG,*m_nMag,*mask,*hist_mask;

extern int width;
	int height;
const int m_nImageWidth=128;

float t_nRes;


Mat m_Mag, m_ANG;

extern "C" void countFeaturesfloat(float *img,float *out,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,
	 int *mask,int *histo_mask,float SVM_bias,float *svm_weight,int svm_count,int Imagewidth,int ImageHeight,
	 float t_fAugRate,vector <CvRect> &t_vTarget);
void CountCell( float * m_pCellFeatures,int t_nY,int t_nX,int width, int t_nWidth, int t_nHeight,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int *mask,int *histo_mask)
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

		m_pCellFeatures[histo_mask[off_y*128+off_x]+ t_nm_nBIN] += p_Mag[i+j*width]*mask[off_x+off_y*128];
		}
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
void CountGrad( Mat t_Image )
{
	//初始化空间m
	m_Mag.create( t_Image.rows, t_Image.cols, CV_32FC1 );
	m_ANG.create( t_Image.rows, t_Image.cols, CV_32FC1 );
	//开始计算
	Mat t_DeltaX;
	t_DeltaX.create( t_Image.rows, t_Image.cols, CV_32FC1 );
	Mat t_DeltaY;
	t_DeltaY.create( t_Image.rows, t_Image.cols, CV_32FC1 );
	Sobel( t_Image, t_DeltaX, CV_32FC1, 1, 0, 1 );
	Sobel( t_Image, t_DeltaY, CV_32FC1, 0, 1, 1 );
	int i, j;
	for ( j = 1; j < t_Image.rows - 1; ++j )
	{
		float *t_pPosDeltaX;		//源数据指针
		t_pPosDeltaX = t_DeltaX.ptr<float>(j);
		float *t_pPosDeltaY;		//源数据指针
		t_pPosDeltaY = t_DeltaY.ptr<float>(j);
		float *t_pPosMag;		//梯度模指针
		t_pPosMag = m_Mag.ptr<float>(j);
		float *t_pPosm_nANG;		//梯度角度指针
		t_pPosm_nANG = m_ANG.ptr<float>(j);
		for ( i = 1; i < t_Image.cols - 1; ++i )
		{
			float t_fDeltaX;
			float t_fDeltaY;
			t_fDeltaX = t_pPosDeltaX[i];
			t_fDeltaY = t_pPosDeltaY[i];
			//t_pPosMag[i] = pow( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY, 0.125f );	//可替换项，归一化
			t_pPosMag[i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );
			t_pPosm_nANG[i] = atan2( t_fDeltaX, t_fDeltaY );
		}
	}
}//CountGrad
void initcell(Mat t_Image)
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
	//预设模板参数
	float t_fCenterX;
	t_fCenterX = m_nImageWidth / 2.0f;		//计算中点坐标
	float t_fCenterY;
	t_fCenterY = m_nImageWidth / 2.0f;
	int i, j;
	
	for ( j = 0; j < m_nImageWidth; ++j )
	{
		for ( i = 0; i < m_nImageWidth; ++i )
		{
			float t_fDeltaX;
			float t_fDeltaY;
			t_fDeltaX = i - t_fCenterX;
			t_fDeltaY = t_fCenterY - j;
			//m_fNormalMat[j * m_nImageWidth + i] = atan2( 0, 1.0f );	
			m_fNormalMat[j * m_nImageWidth + i] = atan2( ( t_fDeltaY ), ( t_fDeltaX ) );		// + PI2;
			m_fMagMat[j * m_nImageWidth + i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );
			if ( m_fNormalMat[j * m_nImageWidth + i] < 0 )
			{
				m_fNormalMat[j * m_nImageWidth + i] = m_fNormalMat[j * m_nImageWidth + i] + PI2;
			}
			if(m_fMagMat[j*m_nImageWidth+i]<64)
				mask[j*m_nImageWidth+i]=1;
			else
				mask[j*m_nImageWidth+i]=0;
			m_nANG[j * m_nImageWidth + i] = (int)( m_fNormalMat[j * m_nImageWidth + i] * 18 / PI2 );
			m_nMag[j*m_nImageWidth+i]=(int)(m_fMagMat[j*m_nImageWidth+i]/10);
			hist_mask[j*m_nImageWidth+i]=m_nANG[j*m_nImageWidth+i]*70+10*m_nMag[j*m_nImageWidth+i];
		}
	}
	CountGrad(t_Image);
}
 

 void countfeature(int off_x,int off_y,int t_width,int t_height)
 {
	 
// CountCell(cpu_out,off_y,off_x,t_width,t_height,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask);
	 CountCell(cpu_out,off_y,off_x,width,t_width,t_height,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask);
	 SmoothCell(cpu_out);
 Countm_nBIN(4,3,10,7,18,cpu_block_out,cpu_out);
 norm2(3,10,18,4,cpu_block_out);
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
	
	//聚类区域
	vector <TargetArea> t_vAreaSeq;
	int i, j;
	
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
void testcell(Mat t_Image,float t_fAugRate,vector <CvRect> &t_vTarget)
{
	initcell(t_Image);
	vector <CvRect> t_vCurRect;
	CvRect t_Rect;
	t_Rect.x = 0;
	t_Rect.y = 0;
	t_Rect.width = m_nImageWidth;
	t_Rect.height = m_nImageWidth;
	int width=t_Image.cols;
	int height=t_Image.rows;
		CvSVM  t_pSVM ;
	string t_sClassFilePath="C:\\Users\\Cyj\\Desktop\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
//float * t_pFeature;
//t_pFeature = new float[my_feature];
	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( my_feature, 1, CV_32FC1 );
		while ( t_Rect.x + m_nImageWidth < t_Image.cols )
	{
		t_Rect.y = 0;

		while ( t_Rect.y + m_nImageWidth < t_Image.rows )
		{
			
			countfeature( t_Rect.x, t_Rect.y, t_Rect.width, t_Rect.height);	//计算特征
			
			int i;
			float t_nRes = 0;
			for ( i = 0; i < 18; ++i )
			{
				memcpy( t_FeatureMat->data.fl, 
						&cpu_block_out[i * 120], 
						(18 - i) * 120 * sizeof ( float ) );

				if ( i > 0 )
				{
					memcpy( &t_FeatureMat->data.fl[(18 - i) * 120], 
							cpu_block_out, 
							i * 120* sizeof ( float ) );
				}
	          t_nRes = t_pSVM.predict( t_FeatureMat ); 
			
				if ( t_nRes > 0.5f )
				{
					break;
				}
			}

			if( t_nRes > 0.5f )
			{
				CvRect t_AddRect;
				t_AddRect.x = (int)( ( t_Rect.x + t_Rect.width * 0.5f ) / t_fAugRate );
				t_AddRect.y = (int)( ( t_Rect.y + t_Rect.height * 0.5f ) / t_fAugRate );

				t_AddRect.width = (int)( m_nImageWidth / t_fAugRate ) - 3;
				t_AddRect.height = t_AddRect.width;

				t_AddRect.width /= 2;
				t_AddRect.height /= 2;

				t_vTarget.push_back( t_AddRect );
			}
			t_Rect.y += m_nSearchStep;
		}

		t_Rect.x += m_nSearchStep;
	}
 delete [] cpu_block_out;
	cvReleaseMat( &t_FeatureMat );


}
void preprocess(Mat t_Image,Mat &t_dst )
{
	cvtColor(t_Image,t_Image,CV_BGR2GRAY);
	imshow("gray",t_Image);
	waitKey(0);
	medianBlur( t_Image, t_Image, 7 );
	imshow("medianBlur",t_Image);
	waitKey(0);
	GaussianBlur( t_Image, t_Image, cvSize( 3, 3 ), 1 );
	imshow("Gaussian",t_Image);
	waitKey(0);
	t_Image.convertTo( t_dst, CV_32F );	
	int i, j;
	for ( j = 0; j < t_Image.rows; ++j )
	{
		float *t_pData; 
		t_pData = t_dst.ptr<float>(j);
		for ( i = 0; i < t_Image.cols; ++i )
		{
			t_pData[i] = sqrt( t_pData[i] );
		}
	}
}
int multiDetect(Mat t_Image,float t_fStartRate,int t_nSearchStep, 
				 iRect *& t_pRect, float t_fStepSize, int t_nResizeStep,int t_nMatchTime)
{
	Mat t_SrcImage,m_TestImage;	
	preprocess(t_Image,t_SrcImage);
	vector <CvRect> t_vTarget;
	for (int  i = 0; i < t_nResizeStep; ++i )
	{
		int t_nNewWidth;
		int t_nNewHeight;
		t_nNewWidth = (int)( t_Image.cols * t_fStartRate );
		t_nNewHeight = (int)( t_Image.rows * t_fStartRate );
		Mat t_NewImage( t_nNewHeight, t_nNewWidth, CV_32FC1 );		//不同尺度图像
		resize( t_SrcImage, t_NewImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		//二次训练图片输出代码
	
			m_TestImage.create( t_nNewHeight, t_nNewWidth, CV_8UC1 );
		
		resize( t_Image, m_TestImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		//二次训练图片输出代码
		//start2=clock();
		//CountGrad( t_NewImage );		//预计算梯度
		//initcell(t_NewImage);
		testcell(t_NewImage,t_fStartRate,t_vTarget);
		
		t_fStartRate += t_fStepSize;		//更新放大率
	}
	return RefineTargetSeq( t_vTarget, t_pRect, t_nMatchTime );
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

int main()
{
	Mat pre;
	FILE *fp;
	string c_name="C:\\Users\\Cyj\\Desktop\\test.jpg";
	Mat src=imread("C:\\Users\\Cyj\\Desktop\\test.jpg",CV_LOAD_IMAGE_UNCHANGED);
	//cvtColor(src,src,CV_BGR2GRAY);
	//imshow("gray",src);
	//waitKey(0);
	//medianBlur( src, src, 7 );
	//imshow("medianBlur",src);
	//waitKey(0);
	//GaussianBlur( src, src, cvSize( 3, 3 ), 1 );
	//imshow("Gaussian",src);
	//waitKey(0);
	int i, j;
	 width=src.cols;
	 height=src.rows;
	/*float *srcIm=(float *)malloc(sizeof(float)*width*height);
	src.convertTo( src, CV_32F );

	for ( j = 0; j < src.rows; ++j )
	{
		float *t_pData; 
		t_pData = src.ptr<float>(j);
		for ( i = 0; i <src.cols; ++i )
		{
			t_pData[i] = sqrt( t_pData[i] );
			srcIm[j*width+i]=t_pData[i];
		}
	}*/
	
	
	CvSVM  t_pSVM ;
	string t_sClassFilePath="C:\\Users\\Cyj\\Desktop\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
	

	int svm_count=t_pSVM.get_var_count();
	float svm_bias=-2.7828561096231148e+000;
	float *svm_weights;
	float t_fAugRate=1;
	svm_weights=(float *)malloc(sizeof(float)*svm_count);
 i=0;
 float res=0;
 
 while(i<svm_count)
 {
	 svm_weights[i]=t_pSVM.get_support_vector(0)[i];
	 i++;
 } 
 /*for(i=0;i<2160;i++)
	 res+=svm_weights[i]*cpu_block_out[i];*/
 //printf("res: %f",res);
 vector <CvRect> t_vTarget;
 
	//countFeaturesfloat(srcIm,dst,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask,svm_bias,
	 //svm_weights,svm_count,width,height,t_fAugRate,t_vTarget);
 
//fp=fopen("G://xx_featANG.txt","w+");
// fp=fopen("G://out.txt","w");
//for (int k=width;k<width*(height-1);k++)
	//fprintf(fp," %f ",dst[k]);
 //fclose(fp);

 IplImage *t_Image=NULL;
 iRect *t_Rect=NULL;
 ListImage tmpListImage;
 tmpListImage.LoadImageFromFile(c_name.c_str());
		cvtList2Ipl(&tmpListImage,t_Image);
		int num=multiDetect(src,0.21f,10,t_Rect ,0.09f,5,5);
			
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