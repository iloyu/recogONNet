#include "sourcev2.h"
#include "stdafx.h"
#include "Header.h"
using namespace cv;


HOG_CPU::HOG_CPU()
{
	long sizem_n=sizeof(int)*128*128;
	long sizem_f=sizeof(float)*128*128;
	long size_mask=sizeof(int)*128*128;
	

	hist_mask=(int*)malloc(size_mask);
	mask=(int*)malloc(size_mask);
	
	m_nANG=(int *)malloc(sizem_n);
	m_nMag=(int *)malloc(sizem_n);
	m_fMagMat=(float *)malloc(sizem_f);
	m_fNormalMat=(float*)malloc(sizem_f);

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
}
HOG_CPU::~HOG_CPU()
{
	/*free(m_nANG);
	free(m_nMag);
	free(m_fMagMat);
	free(m_fNormalMat);
	free(m_MagImage);
	free(m_ANGImage);
	free(cpu_out);
	free(mask);
	free(hist_mask);
	free(cpu_smooth_out);
	free(cpu_block_out);*/
}
void HOG_CPU::CountCell( float * m_pCellFeatures,int t_nY,int t_nX,int width, int t_nWidth, int t_nHeight,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int *mask,int *histo_mask)
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
void HOG_CPU::SmoothCell(float *m_pCellFeatures)
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
void HOG_CPU::Countm_nBIN( int m_nBlobNumb,int m_nCellPerBlob,int m_nBIN,int m_nCellNumb,int m_nANG,float *m_pfFeature,float *m_pCellFeatures)
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
void HOG_CPU::norm2(int m_nCellPerBlob,int m_nBIN,int m_nANG,int m_nBlobNumb,float *m_pfFeature )
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
int HOG_CPU::GetImageList( string t_sPath, vector <string> &t_vFileName )
{
	//清空队列
	t_vFileName.clear();
	int t_nEnd = 0;
	//获取该路径下的所有文件  
	_finddata_t file;
long	long lf;
	string t_sTempPath = t_sPath + "*";
	//输入文件夹路径
	locale loc = locale::global(locale(""));
	lf = (long long )_findfirst( t_sTempPath.c_str(), &file );
	if ( lf == -1 ) 
	{
		locale::global(locale("C"));//还原全局区域设定
		return 0;
	} 
	else  
	{
		while( _findnext( lf, &file ) == 0 ) 
		{
			//输出文件名
			//cout<<file.name<<endl;
			if ( strcmp( file.name, "." ) == 0 || strcmp( file.name, ".." ) == 0 )
			{
				continue;
			}
			string m_strFileExt = strrchr( file.name, '.' );
			if ( m_strFileExt == ".jpg" || m_strFileExt == ".JPG" || m_strFileExt == ".Jpg"
				|| m_strFileExt == ".bmp" || m_strFileExt == ".BMP" || m_strFileExt == ".PNG"|| m_strFileExt == ".png")		//只处理jpg格式文件，如需处理其他格式，在这里添加
			{
				m_strFileExt = t_sPath + file.name;	//生成完整路径+文件名
				t_vFileName.push_back( m_strFileExt );	//添加文件
				t_nEnd++;
			}
		}
	}
	_findclose(lf);
	locale::global(locale("C"));//还原全局区域设定
	return t_nEnd;
}//GetImageList
void HOG_CPU::testaccuaracy(string t_sPosPath, string t_sNegPath, float &t_fPosRate, float &t_fNegRate)
{
	int t_nPosNumber;			//正样本数量
	vector <string> t_vPosFileList;	//用于保存图像名称队列
	t_nPosNumber = GetImageList( t_sPosPath, t_vPosFileList );
	//获取反例图像文件名队列
	int t_nNegNumber;			//负样本数量
	vector <string> t_vNegFileList;	//用于保存图像名称队列
	t_nNegNumber = GetImageList( t_sNegPath, t_vNegFileList );
	//测试代码，用于测试效果
	float t_fSumPos;
	t_fSumPos = 0;
	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( 2160,1, CV_32FC1 );
	float * t_pFeature;
	t_pFeature = new float[2160];
	CvSVM t_pSVM;
		string t_sClassFilePath="..\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
	int i;
	for ( i = 0; i < t_nPosNumber; ++i )
	{
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vPosFileList[i].c_str() );
		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );
		preprocess(t_Image,t_Image);
		initcell(t_Image);
		countfeature(0,0,128,128);
		t_pFeature=cpu_block_out;
		memcpy( t_FeatureMat->data.fl, t_pFeature, 2160 * sizeof ( float ) );
		float t_nS;
		
			t_nS = t_pSVM.predict( t_FeatureMat );
		
		t_fSumPos += t_nS;
	}
	t_fPosRate = t_fSumPos / t_nPosNumber;		//正样本正确率
	float t_fSumNeg;
	t_fSumNeg = (float)t_nNegNumber;
	for ( i = 0; i < t_nNegNumber; ++i )
	{
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vNegFileList[i].c_str() );
		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );
		preprocess(t_Image,t_Image);
		initcell(t_Image);
		countfeature(0,0,128,128);
		t_pFeature=cpu_block_out;
		memcpy( t_FeatureMat->data.fl, t_pFeature, 2160 * sizeof ( float ) );
		float t_nS;
		t_nS = t_pSVM.predict( t_FeatureMat );
		t_fSumNeg -= t_nS;
	}
	t_fNegRate = t_fSumNeg / t_nNegNumber;		//负样本正确率
	delete [] t_pFeature;
	cvReleaseMat( &t_FeatureMat );	//释放资源

	//_LARGE_INTEGER time_start;  //开始时间  
	//_LARGE_INTEGER time_over;   //结束时间  
	//double dqFreq;      //计时器频率  
	//LARGE_INTEGER f;    //计时器频率  
	//QueryPerformanceFrequency(&f);  
	//dqFreq=(double)f.QuadPart;  
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
void HOG_CPU::CountGrad( Mat t_Image )
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

void HOG_CPU:: initcell(Mat t_Image)
{
	Width=t_Image.cols;
	height=t_Image.rows;
	long sizem_p=sizeof(float)*Width*height;
	
	long size_cpu=sizeof(float)*1260;
	
	long size_block=sizeof(float)*2160;
	
	
	m_MagImage=(float*)malloc(sizem_p);
	m_ANGImage=(float*)malloc(sizem_p);
	cpu_out=(float *)malloc(size_cpu);
	cpu_smooth_out=(float*)malloc(size_cpu);
	cpu_block_out=(float *)malloc(size_block);
	
	memset(cpu_out,0,size_cpu);
	
	FILE *fp;
	//预设模板参数
	int i,j;
	CountGrad(t_Image);
	for(i=0;i<height;i++)
	{
		float *ANG=m_ANG.ptr<float>(i);
		float *MAG=m_Mag.ptr<float>(i);
		for(j=0;j<Width;j++)
			{
				m_ANGImage[i*Width+j]=ANG[j];
				m_MagImage[i*Width+j]=MAG[j];
		if(m_MagImage[i*Width+j]<0)
			m_MagImage[i*Width+j]=0;
		if(m_ANGImage[i*Width+j]<-PI)
			m_ANGImage[i*Width+j]=0;
		}
	}

}
 

 void HOG_CPU::countfeature(int off_x,int off_y,int t_width,int t_height)
 {
	 
// CountCell(cpu_out,off_y,off_x,t_width,t_height,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask);
	 MyTimer timer;
       
       
        timer.start();
	 CountCell(cpu_out,off_y,off_x,Width,t_width,t_height,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,mask,hist_mask);
timer.stop();
        //printf("countcell debug Time Elapsed: %lf\n", timer.elapse());
		//getchar();
	
		SmoothCell(cpu_out);	
		
 Countm_nBIN(4,3,10,7,18,cpu_block_out,cpu_out); 

 norm2(3,10,18,4,cpu_block_out);
 
 }

int HOG_CPU::RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime )	
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
int HOG_CPU::testcell(Mat t_Image,float t_fAugRate,vector <CvRect> &t_vTarget)
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
	string t_sClassFilePath="..\\Atomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
	int count=0;
	clock_t start,end;
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
			count++;
			int i;
			float t_nRes = 0;
			MyTimer timer;
			timer.start();
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
  //timer.stop();
			  //cout<<"time elasped"<<timer.elapse()<<endl;
			  //getchar();
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
//			
			t_Rect.y += m_nSearchStep;
		}

		t_Rect.x += m_nSearchStep;
	}
 delete [] cpu_block_out;
	cvReleaseMat( &t_FeatureMat );
	return count;

}
void HOG_CPU::preprocess(Mat t_Image,Mat  &t_dst )
{
	cvtColor(t_Image,t_Image,CV_BGR2GRAY);
	//imshow("gray",t_Image);
	//waitKey(0);
	medianBlur( t_Image, t_Image, 7 );
	//imshow("medianBlur",t_Image);
	//waitKey(0);
	GaussianBlur( t_Image, t_Image, cvSize( 3, 3 ), 1 );
	//imshow("Gaussian",t_Image);
	//waitKey(0);
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
int HOG_CPU::cvtList2Mat( ListImage *SrcImg, Mat & t_Image )
{
	if ( SrcImg->GetImgChannel() == 1 )
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC1 );
		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;
		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
			}
		}
	}
	else if ( SrcImg->GetImgChannel() == 3 )
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC3 );
		int a = t_Image.cols;
		int b = t_Image.rows;
		int c = t_Image.channels();
		CvSize d = t_Image.size();
		int e = SrcImg->GetImgDataSize();
	
		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;
		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
			}
		}
	}
	else
	{
		t_Image.create( SrcImg->GetImgHeight(), SrcImg->GetImgWidth(), CV_8UC3 );
	
		int i, j;
		UCHAR * t_pSrc;
		UCHAR * t_pDst;
		for ( j = 0; j < t_Image.rows; ++j )
		{
			t_pSrc = SrcImg->GetImgBuffer();
			t_pSrc += SrcImg->GetImgLineBytes() * j;
			t_pDst = t_Image.ptr<uchar>(j);
			for ( i = 0; i < t_Image.cols; ++i )
			{
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				*t_pDst = *t_pSrc;
				t_pDst++;
				t_pSrc++;
				
				t_pSrc++;
			}
		}
	}
	
	return 1;
}//cvtList2Mat

int HOG_CPU::multiDetect(ListImage* SrcImage,float t_fStartRate, 
				 iRect *& t_pRect, float t_fStepSize, int t_nResizeStep,int t_nMatchTime)
{
		Mat t_Image;
	cvtList2Mat( SrcImage, t_Image );
	clock_t start2,end2;
	Mat t_SrcImage;	
	preprocess(t_Image,t_SrcImage);
	vector <CvRect> t_vTarget;
	int count=0;
	for (int  i = 0; i < t_nResizeStep; ++i )
	{
		int t_nNewWidth;
		int t_nNewHeight;
		t_nNewWidth = (int)( t_Image.cols * t_fStartRate );
		t_nNewHeight = (int)( t_Image.rows * t_fStartRate );
		Mat t_NewImage( t_nNewHeight, t_nNewWidth, CV_32FC1 );		//不同尺度图像
		resize( t_SrcImage, t_NewImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		
		count+=testcell(t_NewImage,t_fStartRate,t_vTarget);
	
		t_fStartRate += t_fStepSize;		//更新放大率
	}

	return RefineTargetSeq( t_vTarget, t_pRect, t_nMatchTime );
}
void HOG_CPU::release()
{
		
	
}

