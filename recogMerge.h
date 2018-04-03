//*****************************************************************
// 文件名 :						HOGDetector.cpp
// 版本	 :						1.0
// 目的及主要功能 :				用于CHOG特征计算及图像匹配
// 创建日期 :					2016.1.28
// 修改日期 :					
// 作者 :						王征
// 修改者 :						
// 联系方式 :					fiki@seu.edu.cn
//*****************************************************************/
///////////////////////////////////////////////////////
////////////////////////include////////////////////////
///////////////////////////////////////////////////////
#include "StdAfx.h"
#include "RHOG.h"
#include "Markup.h"		//用于输出xml文件
#include "time.h"
#include <windows.h> 
/*****************************************************************
Defines
*****************************************************************/
//none
/*****************************************************************
Global Variables
*****************************************************************/
int g_nPosImageNumber = 1;
int g_nNegImageNumber = 1;
/*****************************************************************
Global Function
*****************************************************************/
/*****************************************************************
							RHOG类定义
*****************************************************************/ 
/*****************************************************************
Name:			RHOG
Inputs:
	none.
Return Value:
	none.
Description:	默认构造函数
*****************************************************************/
RHOG::RHOG(void)
{
	//初始化参数
	m_nANG = ANG;				//分多少个角度方向，必须是偶数
	m_nCellNumb = CellNumb;		//每个角度方向分多少个cell
	m_nCellPerBlob = CellPerBlob;		//每三个cell一个blob
	m_nBlobNumb = m_nCellNumb - m_nCellPerBlob;		//每个角度方向多少个m_nBlobNumb
	m_nBIN = 10;				//每个cell分多少个梯度方向
	m_nImageWidth = ImageWidth;	//图像大小
	m_nSearchStep = SearchStep;	//搜索步长
	m_nClassType = SVMC;		//分类器
	m_bSym = Sym;				//是否需要对称特征
	m_bRSC = RSC;				//是否对Cell进行角度间的平滑
	m_bDSC = DSC;				//是否对Cell进行角度间的平滑
	m_nFilterSize = FilterSize;	//默认中值滤波器模板大小
	m_nMatchTime = MatchTime;	//计算结果时模板重叠的次数
	m_bSavePosPatch = false;	//是否保留检测中所截取的正例子图
	m_bSaveNegPatch = false;	//是否保留检测中所截取的反例子图
	m_sPosPatchPath = "E:\\Train\\Pos\\";	//检测中所截取的正例子图保存路径
	m_sNegPatchPath = "E:\\Train\\Neg\\";	//检测中所截取的正例子图保存路径
	//初始化空间指针及相关参数
	m_pfFeature = NULL;			//特征向量
	m_pCellFeatures = NULL;		//Cell特征向量
	m_pClassifier=NULL;			//分类器初始化为空
	if ( m_bSym )
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN + m_nANG / 2 * m_nCellNumb;		//特征总数
	}
	else
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//特征总数 18*4*3*10
	}
	m_nANGWidth = m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//每个方向的特征数 4*3*10
	//开辟初始化空间
	m_fNormalMat = new float[m_nImageWidth * m_nImageWidth];
	m_fMagMat = new float[m_nImageWidth * m_nImageWidth];
	m_nANGle = new int[m_nImageWidth * m_nImageWidth];
	m_nMag=new int[m_nImageWidth * m_nImageWidth];
	mask=new int[m_nImageWidth*m_nImageWidth];
	histo_mask=new int [m_nImageWidth*m_nImageWidth];
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
			m_nANGle[j * m_nImageWidth + i] = (int)( m_fNormalMat[j * m_nImageWidth + i] * m_nANG / PI2 );
			m_nMag[j*m_nImageWidth+i]=(int)(m_fMagMat[j*m_nImageWidth+i]/10);
			histo_mask[j*m_nImageWidth+i]=m_nANGle[j*m_nImageWidth+i]*70+10*m_nMag[j*m_nImageWidth+i];
		}
	}
}//RHOG
/*****************************************************************
Name:			~HOGDetector
Inputs:
	none.
Return Value:
	none.
Description:	析构函数
*****************************************************************/
RHOG::~RHOG(void)
{
	Clear();	//释放空间
	if ( m_fNormalMat != NULL )
	{
		delete [] m_fNormalMat;
		m_fNormalMat = NULL;
	}
	if ( m_fMagMat != NULL )
	{
		delete [] m_fMagMat;
		m_fMagMat = NULL;
	}
	if ( m_nANGle != NULL )
	{
		delete [] m_nANGle;
		m_nANGle = NULL;
	}
	if ( mask != NULL )
	{
		delete [] mask;
		mask = NULL;
	}
	if (histo_mask!= NULL )
	{
		delete [] histo_mask;
		histo_mask = NULL;
	}
	if ( m_nMag != NULL )
	{
		delete [] m_nMag;
		m_nMag = NULL;
	}
	//关闭分类器
	if (m_pClassifier!=NULL)
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;
		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}
}//~RHOG(void)
/*****************************************************************
Name:			GetPar
Inputs:
	RHOGPar &t_Par - 返回读出的参数
Return Value:
	none
Description:	读出当前系统参数
*****************************************************************/
void RHOG::GetPar( RHOGPar &t_Par )
{
	t_Par.m_nANG = m_nANG;		//分多少个角度方向，必须是偶数
	t_Par.m_nCellNumb = m_nCellNumb;		//每个角度方向分多少个cell
	t_Par.m_nCellPerBlob = m_nCellPerBlob;	//每三个cell一个blob
	t_Par.m_nBIN = m_nBIN;		//每个cell分多少个梯度方向
	t_Par.m_nImageWidth = m_nImageWidth;//图像大小
	t_Par.m_nClassType = m_nClassType;	//分类器
	t_Par.m_bSym = m_bSym;		//是否需要对称特征
	t_Par.m_bRSC = m_bRSC;		//是否对Cell进行角度间的平滑
	t_Par.m_bDSC = m_bDSC;		//是否对Cell进行角度间的平滑
	t_Par.m_nFilterSize = m_nFilterSize;	//图像预处理中值滤波器模板大小，必须是奇数
}//GetPar
/*****************************************************************
Name:			SetPar
Inputs:
	RHOGPar t_Par - 待设置的分析参数
Return Value:
	1 - 保存成功
	<0 - 保存错误
Description:	设置参数，清空当前开辟空间，并根据参数开辟空间
*****************************************************************/
int RHOG::SetPar( RHOGPar t_Par )
{
	//判断合法性
	if ( t_Par.m_nANG < 4 
		|| t_Par.m_nCellNumb < 5
		|| t_Par.m_nCellNumb > 20
		|| t_Par.m_nCellPerBlob >=  t_Par.m_nCellNumb - 2
		|| t_Par.m_nBIN < 5
		|| t_Par.m_nBIN > 18
		|| t_Par.m_nImageWidth > 256
		|| t_Par.m_nImageWidth < 32
		|| t_Par.m_nClassType > SVMC
		|| t_Par.m_nClassType < Adaboost
		|| t_Par.m_nFilterSize % 2 != 1
		|| t_Par.m_nFilterSize > 21 )
	{
		return ParIllegal;
	}
	//复制参数
	m_nANG = t_Par.m_nANG;		//分多少个角度方向，必须是偶数
	m_nCellNumb = t_Par.m_nCellNumb;		//每个角度方向分多少个cell
	m_nCellPerBlob = t_Par.m_nCellPerBlob;	//每三个cell一个blob
	m_nBlobNumb = m_nCellNumb - m_nCellPerBlob;		//每个角度方向多少个m_nBlobNumb
	m_nBIN = t_Par.m_nBIN;		//每个cell分多少个梯度方向
	m_nImageWidth = t_Par.m_nImageWidth;//图像大小
	m_nClassType = t_Par.m_nClassType;	//分类器
	m_bSym = t_Par.m_bSym;		//是否需要对称特征
	m_bRSC = t_Par.m_bRSC;		//是否对Cell进行角度间的平滑
	m_bDSC = t_Par.m_bDSC;		//是否对Cell进行角度间的平滑
	m_nFilterSize = t_Par.m_nFilterSize;//图像预处理中值滤波器模板大小
	//清理现有空间
	Clear();
	if ( m_fNormalMat != NULL )
	{
		delete [] m_fNormalMat;
		m_fNormalMat = NULL;
	}
	if ( m_fMagMat != NULL )
	{
		delete [] m_fMagMat;
		m_fMagMat = NULL;
	}
	if ( m_nANGle != NULL )
	{
		delete [] m_nANGle;
		m_nANGle = NULL;
	}
	//关闭分类器
	if (m_pClassifier!=NULL)
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;
		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}
	//开辟空间
	if ( m_bSym )
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN + m_nANG / 2 * m_nCellNumb;		//特征总数
	}
	else
	{
		m_nFeatureNumber = m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN;		//特征总数
	}
	m_nANGWidth = m_nBlobNumb * m_nCellPerBlob * m_nBIN;					//每个方向的特征数
	m_fNormalMat = new float[m_nImageWidth * m_nImageWidth];	
	m_fMagMat = new float[m_nImageWidth * m_nImageWidth];
	m_nANGle = new int[m_nImageWidth * m_nImageWidth];
	mask=new int[m_nImageWidth*m_nImageWidth];
	m_nMag=new int[m_nImageWidth * m_nImageWidth];
	histo_mask=new int [m_nImageWidth*m_nImageWidth];
	
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
			m_fNormalMat[j * m_nImageWidth + i] = atan2( 0, 1.0f );	
			m_fNormalMat[j * m_nImageWidth + i] = atan2( ( t_fDeltaY ), ( t_fDeltaX ) );		// + PI2;
			m_fMagMat[j * m_nImageWidth + i] = sqrt( t_fDeltaX * t_fDeltaX + t_fDeltaY * t_fDeltaY );
			if ( m_fNormalMat[j * m_nImageWidth + i] < 0 )
			{
				m_fNormalMat[j * m_nImageWidth + i] = m_fNormalMat[j * m_nImageWidth + i] + PI2;
			}
			if(m_fMagMat[j*m_nImageWidth+i]<64)
				mask[j*m_nImageWidth+i]=1;
			else
			{
				mask[j*m_nImageWidth+i]=0;
			}
			m_nANGle[j * m_nImageWidth + i] = (int)( m_fNormalMat[j * m_nImageWidth + i] * m_nANG / PI2 );
			m_nMag[j*m_nImageWidth+i]=(int)(m_fMagMat[j*m_nImageWidth+i]/10);
			histo_mask[j*m_nImageWidth+i]=m_nANGle[j*m_nImageWidth+i]*70+10*m_nMag[j*m_nImageWidth+i];
			
		}
	}
	return 1;
}//SetPar
/*****************************************************************
Name:			SaveClassifier
Inputs:
	string t_sClassFilePath - 保存的分类器路径及文件名
Return Value:
	1 - 保存成功
	<0 - 保存错误
Description:	保存当前分类器至文件，文件名应以xml结尾。
*****************************************************************/
int RHOG::SaveClassifier( string t_sClassFilePath )
{
	//写入分类器
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	switch ( m_nClassType )
	{
	case Adaboost:	((CvBoost *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	case Rtree:		((CvRTrees *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	case SVMC:		((CvSVM *)m_pClassifier)->save( t_sClassFilePath.c_str() );break;
	}
	locale loc = locale::global(locale(""));
		
	ifstream ifs( t_sClassFilePath.c_str());
	if( !ifs )
	{
		locale::global(locale("C"));//还原全局区域设定
		return ClassifierSaveFailed;
	}
	//写入参数信息
	string t_sKey;
	string t_sOut;
	CMarkup t_XML;  
	t_XML.Load( t_sClassFilePath.c_str() );   
	t_XML.AddElem( "ClassifierPar" );
	t_XML.IntoElem();
	t_XML.AddElem( "Par" );
	char p[32]={0,};//初始化临时字符串
	t_sKey = "ANG";
	sprintf( p,"%d", m_nANG );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "CellNumb";
	sprintf( p,"%d", m_nCellNumb );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "CellPerBlob";
	sprintf( p,"%d", m_nCellPerBlob );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "BIN";
	sprintf( p,"%d", m_nBIN );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "ImageWidth";
	sprintf( p,"%d", m_nImageWidth );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "ClassType";
	sprintf( p,"%d", m_nClassType );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "Sym";
	sprintf( p,"%d", m_bSym );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "RSC";
	sprintf( p,"%d", m_bRSC );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "DSC";
	sprintf( p,"%d", m_bDSC );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_sKey = "FilterSize";
	sprintf( p,"%d", m_nFilterSize );
	t_sOut = p;
	t_XML.AddAttrib( t_sKey.c_str(), t_sOut.c_str() );
	t_XML.OutOfElem();
	t_XML.Save( t_sClassFilePath.c_str() );
	locale::global(locale("C"));//还原全局区域设定
	return 1;
}//SaveClassifier
/*****************************************************************
Name:			LoadClassifier
Inputs:
	string t_sClassFilePath - 读取的分类器路径
Return Value:
	1 - 读取成功
	<0 - 读取错误
Description:	读取的分类器，读入参数后，会根据读取的
				分类器参数，重新开辟空间
*****************************************************************/
int RHOG::LoadClassifier( string t_sClassFilePath )
{
	locale loc = locale::global(locale(""));
	ifstream ifs( t_sClassFilePath.c_str() );
	if( !ifs )
	{
		locale::global(locale("C"));//还原全局区域设定
		return ClassifierSaveFailed;
	}
	//读取XML信息
	RHOGPar t_RHOGPar;
	{
		bool t_bFind;
		CMarkup t_XML;  
		t_bFind = t_XML.Load( t_sClassFilePath );  
		if ( !t_bFind )
		{
			locale::global(locale("C"));//还原全局区域设定
			return ClassifierFileNotExist;
		}
		string t_sKey;
		string t_sIn;
		t_XML.ResetMainPos();
		t_XML.FindElem( "ClassifierPar" );    //UserInfo
		while( t_XML.FindChildElem("Par") )
		{
			t_sIn = t_XML.GetChildAttrib( "ANG" );
			t_RHOGPar.m_nANG = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "CellNumb" );
			t_RHOGPar.m_nCellNumb = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "CellPerBlob" );
			t_RHOGPar.m_nCellPerBlob = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "BIN" );
			t_RHOGPar.m_nBIN = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "ImageWidth" );
			t_RHOGPar.m_nImageWidth = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "ClassType" );
			t_RHOGPar.m_nClassType = atoi( t_sIn.c_str() );
			t_sIn = t_XML.GetChildAttrib( "Sym" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bSym = false;
			}
			else
			{
				t_RHOGPar.m_bSym = true;
			}
			t_sIn = t_XML.GetChildAttrib( "RSC" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bRSC = false;
			}
			else
			{
				t_RHOGPar.m_bRSC = true;
			}
			t_sIn = t_XML.GetChildAttrib( "DSC" );
			if ( t_sIn == "0" )
			{
				t_RHOGPar.m_bDSC = false;
			}
			else
			{
				t_RHOGPar.m_bDSC = true;
			}
			t_sIn = t_XML.GetChildAttrib( "FilterSize" );
			t_RHOGPar.m_nFilterSize = atoi( t_sIn.c_str() );
		}
		t_XML.FindElem( "ClassifierPar" );    //UserInfo
		t_XML.RemoveElem();
		t_XML.Save( t_sClassFilePath );
	}
	SetPar( t_RHOGPar );		//重新设置参数
	//读取分类器
	switch( m_nClassType )
	{
	case Adaboost:	{ 
		CvBoost * t_pBoost = new CvBoost;
		t_pBoost->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pBoost;
		break;
					}
	case Rtree:		{ 
		CvRTrees * t_pRTrees = new CvRTrees;
		t_pRTrees->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pRTrees;
		break;
					}
	case SVMC:		{ 
		CvSVM * t_pSVM = new CvSVM;
		t_pSVM->load( t_sClassFilePath.c_str() );
		m_pClassifier = (void *)t_pSVM;
		break;
					}
	default: return WrongClassifierType;
		break;
	}
	//重新写入信息
	//写入参数信息
	{
		string t_sKey;
		string t_sOut;
		CMarkup t_XML;  
		t_XML.Load( t_sClassFilePath );   
		t_XML.AddElem( "ClassifierPar" );
		t_XML.IntoElem();
		t_XML.AddElem( "Par" );
		char p[32]={0,};//初始化临时字符串
		t_sKey = "ANG";
		sprintf( p,"%d", m_nANG );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "CellNumb";
		sprintf( p,"%d", m_nCellNumb );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "CellPerBlob";
		sprintf( p,"%d", m_nCellPerBlob );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "BIN";
		sprintf( p,"%d", m_nBIN );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "ImageWidth";
		sprintf( p,"%d", m_nImageWidth );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "ClassType";
		sprintf( p,"%d", m_nClassType );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "Sym";
		sprintf( p,"%d", m_bSym );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "RSC";
		sprintf( p,"%d", m_bRSC );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "DSC";
		sprintf( p,"%d", m_bDSC );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_sKey = "FilterSize";
		sprintf( p,"%d", m_nFilterSize );
		t_sOut = p;
		t_XML.AddAttrib( t_sKey, t_sOut );
		t_XML.OutOfElem();
		t_XML.Save( t_sClassFilePath );
	}
	locale::global(locale("C"));//还原全局区域设定
	return 1;
}//LoadClassifier
/*****************************************************************
Name:			SetSavePatchImage
Inputs:
	bool t_bPosSave - 是否保存正样本子图
	bool t_bNegSave - 是否保存负样本子图
	string t_sPosPath - 正样本子图保存路径
	string t_sNegPath - 负样本子图保存路径
Return Value:
	none
Description:	设置是否保存正/负测试分割子图（用于分析和二次训练）
*****************************************************************/
void RHOG::SetSavePatchImage( bool t_bPosSave, bool t_bNegSave, string t_sPosPath, string t_sNegPath )
{
	m_bSavePosPatch = t_bPosSave;
	m_bSaveNegPatch = t_bNegSave;
	m_sPosPatchPath = t_sPosPath;
	m_sNegPatchPath = t_sNegPath;
}//SetSavePatchImage
/*****************************************************************
Name:			Training
Inputs:
	string t_sPosPath - 正样本的路径
	string t_sNegPath - 负样本的路径
Return Value:
	int - <0 错误代码
		  1  正确返回
Description:	训练分类器，训练结果需SaveClassifier函数保存
*****************************************************************/
int RHOG::Training( string t_sPosPath, string t_sNegPath )
{
	//获取正例图像文件名队列
	int t_nPosNumber;			//正样本数量
	vector <string> t_vPosFileList;	//用于保存图像名称队列
	t_nPosNumber = GetImageList( t_sPosPath, t_vPosFileList );
	if ( t_nPosNumber < 20 )
	{
		return NotEnoughSampleForTrainPos;
	}
	//获取反例图像文件名队列
	int t_nNegNumber;			//负样本数量
	vector <string> t_vNegFileList;	//用于保存图像名称队列
	t_nNegNumber = GetImageList( t_sNegPath, t_vNegFileList );
	if ( t_nNegNumber < 20 )
	{
		return NotEnoughSampleForTrainNeg;
	}
	//开辟特征存储空间和标签存储空间
	int t_nSampleNumber;		//总样本数量
	t_nSampleNumber = t_nPosNumber + t_nNegNumber * ( m_nANG / 2 );
	Clear();
	InitFeatures();			//开辟特征空间
	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( t_nSampleNumber, m_nFeatureNumber, CV_32FC1 );
	CvMat * t_ResponseMat;
	t_ResponseMat = cvCreateMat( t_nSampleNumber, 1, CV_32FC1 );
	//关闭已打开的分类器
	if ( m_pClassifier != NULL )
	{
		CvBoost *t_Booster;
		CvRTrees *t_RTrees;
		CvSVM *t_SVM;
		if ( m_nClassType == Rtree )
		{
			t_RTrees=(CvRTrees*)m_pClassifier;
			delete t_RTrees;
			m_pClassifier=NULL;
		}
		else if ( m_nClassType == Adaboost )
		{
			t_Booster=(CvBoost*)m_pClassifier;
			delete t_Booster;
			m_pClassifier=NULL;
		}
		else 
		{
			t_SVM=(CvSVM*)m_pClassifier;
			delete t_SVM;
			m_pClassifier=NULL;
		}
	}
	//计算正例特征队列
	float * t_pFeature;
	t_pFeature = new float[m_nFeatureNumber];
	int i,j;
	for ( i = 0; i < t_nPosNumber; ++i )
	{
		//计算特征
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vPosFileList[i].c_str() );
		t_Img.ConvertToGreyImg();
		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );
		CountFeatureFromImg( t_Image, t_pFeature );	//计算特征
		float * t_pPos;
		t_pPos = &t_FeatureMat->data.fl[t_FeatureMat->width * i];
		memcpy( t_pPos, t_pFeature, m_nFeatureNumber * sizeof( float ) );
		t_ResponseMat->data.fl[i] = 1;
	}
	//计算反例特征队列
	for ( i = 0; i < t_nNegNumber; ++i )
	{
		//计算特征
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vNegFileList[i].c_str() );
		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );
		CountFeatureFromImg( t_Image, t_pFeature );
		//反例旋转使用
		for ( j = 0; j < ( m_nANG / 2 ); j++ )
		{
			float * t_pPos;
			t_pPos = &t_FeatureMat->data.fl[t_FeatureMat->width * ( t_nPosNumber + ( i * ( m_nANG / 2 ) + j ) )];
			memcpy( t_pPos, 
					&t_pFeature[j * 2 * m_nANGWidth], 
					(m_nANG - j * 2) * m_nANGWidth * sizeof ( float ) );
			if ( j > 0 )
			{
				memcpy( &t_pPos[(m_nANG - j * 2) * m_nANGWidth], 
						t_pFeature, 
						j * 2 * m_nANGWidth * sizeof ( float ) );
			}
			if ( m_bSym )
			{
				memcpy( &t_pPos[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
						&t_pFeature[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
						m_nANG * m_nCellNumb * sizeof ( float ) / 2 );
			}
			t_ResponseMat->data.fl[t_nPosNumber + i * ( m_nANG / 2 ) + j] = 0;
		}
	}
	delete [] t_pFeature;
	//设置分类器参数
	if ( m_nClassType == Rtree )
	{
		float priors[] = {1,1};  // weights of each classification for classes
		CvRTParams params = CvRTParams(20, // max depth
										50, // min sample count
										0, // regression accuracy: N/A here
										false, // compute surrogate split, no missing data
										15, // max number of categories (use sub-optimal algorithm for larger numbers)
										priors, // the array of priors
										false,  // calculate variable importance
										50,       // number of variables randomly selected at node and used to find the best split(s).
										100,     // max number of trees in the forest
										0.01f,                // forest accuracy
										CV_TERMCRIT_ITER |    CV_TERMCRIT_EPS // termination cirteria
										);
		CvRTrees *t_pRTree;
		t_pRTree = new CvRTrees;
		//开始训练
		t_pRTree->train( t_FeatureMat, CV_ROW_SAMPLE, t_ResponseMat,
			0, 0, 0, 0, params );
		//保存结果
		m_pClassifier = (void *)t_pRTree;
	}
	else if ( m_nClassType == Adaboost )
	{
		CvBoost *t_pBooster;
		t_pBooster = new CvBoost;
		CvBoostParams t_BoostParams( CvBoost::REAL, 150, 0, 1, false, 0 );
		//开始训练
		t_pBooster->train( t_FeatureMat, CV_ROW_SAMPLE, t_ResponseMat, 0, 0, 0, 0, t_BoostParams, false );
		//保存结果
		m_pClassifier = (void *)t_pBooster;
	}
	else
	{
		//设置支持向量机的参数  
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;//SVM类型：使用C支持向量机
		params.kernel_type = CvSVM::LINEAR;//核函数类型：线性
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//终止准则函数：当迭代次数达到最大值时终止
		//训练SVM
		//建立一个SVM类的实例
		CvSVM *t_pSVM;
		t_pSVM = new CvSVM;
		//训练模型，参数为：输入数据、响应、XX、XX、参数（前面设置过）
		t_pSVM->train( t_FeatureMat, t_ResponseMat, 0, 0, params );  
		//保存结果
		m_pClassifier = (void *)t_pSVM;
	}
	cvReleaseMat( &t_FeatureMat );
	cvReleaseMat( &t_ResponseMat );
	return 1;
}//Training
/*****************************************************************
Name:			Test
Inputs:
	string t_sPosPath - 正样本路径
	string t_sNegPath - 负样本路径
	float &t_fPosRate - 正样本正确率
	float &t_fNegRate - 负样本正确率
Return Value:
	int - <0 错误代码
		  1  正确返回
Description:	测试分类器正确率，以子图为测试图片，
				不同于正常的目标检测。
*****************************************************************/
int RHOG::Test( string t_sPosPath, string t_sNegPath, float &t_fPosRate, float &t_fNegRate )
{
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	//获取正例图像文件名队列
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
	t_FeatureMat = cvCreateMat( m_nFeatureNumber, 1, CV_32FC1 );
	float * t_pFeature;
	t_pFeature = new float[m_nFeatureNumber];
	int i;
	for ( i = 0; i < t_nPosNumber; ++i )
	{
		ListImage t_Img;
		t_Img.LoadImageFromFile( t_vPosFileList[i].c_str() );
		Mat t_Image;
		cvtList2Mat( &t_Img, t_Image );
		CountFeatureFromImg( t_Image, t_pFeature );
		memcpy( t_FeatureMat->data.fl, t_pFeature, m_nFeatureNumber * sizeof ( float ) );
		float t_nS;
		if ( m_nClassType == Rtree )
		{
			t_nS = ((CvRTrees*)m_pClassifier)->predict( t_FeatureMat );
		}
		else if ( m_nClassType == Adaboost )
		{
			t_nS = ((CvBoost*)m_pClassifier)->predict( t_FeatureMat );
		}
		else
		{
			t_nS = ((CvSVM*)m_pClassifier)->predict( t_FeatureMat );
		}
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
		CountFeatureFromImg( t_Image, t_pFeature );
		memcpy( t_FeatureMat->data.fl, t_pFeature, m_nFeatureNumber * sizeof ( float ) );
		float t_nS;
		if ( m_nClassType == Rtree )
		{
			t_nS = ((CvRTrees*)m_pClassifier)->predict( t_FeatureMat );
		}
		else if ( m_nClassType == Adaboost )
		{
			t_nS = ((CvBoost*)m_pClassifier)->predict( t_FeatureMat );
		}
		else
		{
			t_nS = ((CvSVM*)m_pClassifier)->predict( t_FeatureMat );
		}
		t_fSumNeg -= t_nS;
	}
	t_fNegRate = t_fSumNeg / t_nNegNumber;		//负样本正确率
	delete [] t_pFeature;
	cvReleaseMat( &t_FeatureMat );	//释放资源
	return 1;
}//Test
/*****************************************************************
Name:			SearchTarget
Inputs:
	ListImage *SrcImg - 待搜索的图像
	iRect *& t_pRect - 目标队列
	float t_fStartRat - 开始匹配时的缩放比例，初始化为0.5
	float t_fStepSize - 每次搜索时图像缩放的比例改变量，初始化为0.1
	int t_nResizeStep - 匹配时图像缩放的次数
	int t_nMatchTime - 匹配时的重叠阈值
	int t_nSearchStep - 匹配时滑动窗的滑动步长
Return Value:
	int - >0 目标数量
		  <0 对应错误代码
Description:	在图像中检索目标
*****************************************************************/
int RHOG::SearchTarget( ListImage *SrcImg, iRect *& t_pRect, 
						float t_fStartRate, float t_fStepSize, int t_nResizeStep,
						int t_nMatchTime,
						int t_nSearchStep )
{
	//判断输入
	if ( SrcImg->GetImgWidth() < 70 || SrcImg->GetImgHeight() < 70 )
	{
		return InputImageError;
	}
	if ( SrcImg->GetImgDataType() != uint_8 )
	{
		return InputImageNotSupport;
	}
	if ( t_nSearchStep > 0 )
	{
		m_nSearchStep = t_nSearchStep;
	}
	if ( t_nMatchTime <= 0 )
	{
		t_nMatchTime = m_nMatchTime;
	}
	if ( t_fStartRate <= 0.1f || t_fStartRate > 3.0f || fabs( t_fStepSize ) < 0.01f 
		|| t_fStartRate + t_fStepSize * t_nResizeStep > 3.0f
		|| t_fStartRate + t_fStepSize * t_nResizeStep <= 0.1f )
	{
		return ParIllegal;
	}
	//判断分类器是否存在
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	clock_t Teststart,End1,End2,end3,end4,start2,start3,start4,start5;
	CString out;
	
	Mat t_Image;
	cvtList2Mat( SrcImg, t_Image );
	Teststart=clock();
	//初始化
	Clear();
	End1=clock();
	out.Format("clear():  %lf ms",double(End1-Teststart));
	//AfxMessageBox(out);
	start4=clock();
	InitFeatures();			//开辟特征空间
	End2=clock();
	out.Format("InitFeatures():  %lf ms",double(End2-start4));
	//AfxMessageBox(out);
	//确定缩放步数
	int t_nRectNumber;
	t_nRectNumber = 0;
	//图像预处理
	Mat t_SrcImage;		//用于存放32位图像
	start5=clock();
	PreProcessImage( t_Image, t_SrcImage );
	end3=clock();
	out.Format("PreProcessImage:  %lf ms",double(end3-start5));
	//AfxMessageBox(out);
	//不同尺度下搜索
	vector <CvRect> t_vTarget;
	int i;
	for ( i = 0; i < t_nResizeStep; ++i )
	{
		int t_nNewWidth;
		int t_nNewHeight;
		t_nNewWidth = (int)( t_Image.cols * t_fStartRate );
		t_nNewHeight = (int)( t_Image.rows * t_fStartRate );
		Mat t_NewImage( t_nNewHeight, t_nNewWidth, CV_32FC1 );		//不同尺度图像
		resize( t_SrcImage, t_NewImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		//二次训练图片输出代码
		if ( t_Image.channels() == 3 )
		{
			m_TestImage.create( t_nNewHeight, t_nNewWidth, CV_8UC3 );
		}
		else
		{
			m_TestImage.create( t_nNewHeight, t_nNewWidth, CV_8UC1 );
		}
		resize( t_Image, m_TestImage, Size( t_nNewWidth, t_nNewHeight ), 0, 0, CV_INTER_LINEAR );
		//二次训练图片输出代码
		start2=clock();
		CountGrad( t_NewImage );		//预计算梯度
		end3=clock();
	out.Format("countGradImage:  %lf ms",double(end3-start2));
	//AfxMessageBox(out);
	start3=clock();
	SearchTargetPerImg( t_NewImage, t_fStartRate, t_vTarget );
	end4=clock();
	out.Format("SearchTargetPerImage:  %lf ms",double(end4-start3));
	//AfxMessageBox(out);
		t_fStartRate += t_fStepSize;		//更新放大率
	}
	
	return RefineTargetSeq( t_vTarget, t_pRect, t_nMatchTime );
}//SearchTarget
/*****************************************************************
Name:			SearchTargetPerImg
Inputs:
	Mat t_Image - 待分析图像
	float t_fAugRate - 图像放大比例
	vector <CvRect> &t_vTarget - 返回的结果队列
Return Value:
	int - 1 正常返回
		  <0 对应错误代码
Description:	在单张图像中检索目标
*****************************************************************/
int RHOG::SearchTargetPerImg( Mat t_Image, float t_fAugRate, vector <CvRect> &t_vTarget )
{
	//判断分类器是否存在
	if ( m_pClassifier == NULL )
	{
		return ClassifierNotExist;
	}
	double count=0;
	//初步扫描
	vector <CvRect> t_vCurRect;
	CvRect t_Rect;
	t_Rect.x = 0;
	t_Rect.y = 0;
	t_Rect.width = m_nImageWidth;
	t_Rect.height = m_nImageWidth;
	float * t_pFeature,*t_oFeature;
	t_pFeature = new float[m_nFeatureNumber];//2160
	
	CvMat * t_FeatureMat;
	t_FeatureMat = cvCreateMat( m_nFeatureNumber, 1, CV_32FC1 );

	int width=t_Image.cols;
	int height=t_Image.rows;
	
	//CountGrad(t_Image);
	FILE *fp;
	float *p_Mag,*p_ANG;
	
	long sizem_p=sizeof(float)*width*height;
	
	p_Mag=(float*)malloc(sizem_p);
	p_ANG=(float*)malloc(sizem_p);
int i,j;

	for( i=0;i<height;i++)
	{
		float *p_temp_Mag=m_MagImage.ptr<float>(i);
		float *p_temp_ANg=m_ANGImage.ptr<float>(i);
		for( j=0;j<width;j++)
			{ 
			
				p_Mag[i*width+j]=p_temp_Mag[j];
		      p_ANG[i*width+j]=p_temp_ANg[j];
			  if(p_Mag[i*width+j]<0)
				  p_Mag[i*width+j]=0;
			  if(p_ANG[i*width+j]<-4)
				  p_ANG[i*width+j]=0;
			  if(p_ANG[i*width+j]<0)
				  p_ANG[i*width+j]+=PI2;
		}

	}
	/*fp=fopen("G://t_Mag.txt","w");
	for(i=0;i<height;i++)
		{for (j=0;j<width;j++)
		fprintf(fp," %f",p_Mag[i*width+j]);
	fprintf(fp,"\n");}
	fclose(fp);
	fp=fopen("G://t_ANg.txt","w");
	for(i=0;i<height;i++)
		{for (j=0;j<width;j++)
		fprintf(fp," %f",p_ANG[i*width+j]);
	fprintf(fp,"\n");}
	fclose(fp);
	fp=fopen("G://t_fANg.txt","w");
	for(i=0;i<128;i++)
		{for (j=0;j<128;j++)
		fprintf(fp," %f",m_fNormalMat[i*128+j]);
	fprintf(fp,"\n");}
	fclose(fp);
	fp=fopen("G://t_fMag.txt","w");
	for(i=0;i<128;i++)
		{for (j=0;j<128;j++)
		fprintf(fp," %f",m_fMagMat[i*128+j]);
	fprintf(fp,"\n");}
	fclose(fp);*/
	/*fp=fopen("G://t_mask.txt","w");
	for(i=0;i<128;i++)
		{for (j=0;j<128;j++)
		fprintf(fp," %d",mask[i*128+j]);
	fprintf(fp,"\n");}
	fclose(fp);
	fp=fopen("G://t_histo_mask.txt","w");
	for(i=0;i<128;i++)
		{for (j=0;j<128;j++)
		fprintf(fp," %d",histo_mask[i*128+j]);
	fprintf(fp,"\n");}
	fclose(fp);*/
	while ( t_Rect.x + m_nImageWidth < t_Image.cols )
	{
		t_Rect.y = 0;
		while ( t_Rect.y + m_nImageWidth < t_Image.rows )
		{
			
	//		CountFeature( t_Rect.x, t_Rect.y, t_Rect.width, t_Rect.height, t_pFeature );	//计算特征
	//		fp=fopen("G://re_feat.txt","w");
	//for (j=0;j<2160;j++)
	//	fprintf(fp," %f",t_pFeature[j]);
	//fclose(fp);

			countFeaturesfloat(t_pFeature,m_fNormalMat,m_fMagMat,p_ANG,
				p_Mag,width,height,mask,histo_mask,t_Rect.x,t_Rect.y);
				/*fp=fopen("G://my_feat.txt","w");
	for (j=0;j<2160;j++)
		fprintf(fp," %f",t_pFeature[j]);
	fclose(fp);*/
			int i;
			float t_nRes = 0;
			for ( i = 0; i < m_nANG; ++i )
			{
				memcpy( t_FeatureMat->data.fl, 
						&t_pFeature[i * m_nANGWidth], 
						(m_nANG - i) * m_nANGWidth * sizeof ( float ) );
				if ( i > 0 )
				{
					memcpy( &t_FeatureMat->data.fl[(m_nANG - i) * m_nANGWidth], 
							t_pFeature, 
							i * m_nANGWidth * sizeof ( float ) );
				}
				if ( m_bSym )
				{
					memcpy( &t_FeatureMat->data.fl[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
							&t_pFeature[m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN],
							m_nANG * m_nCellNumb * sizeof ( float ) / 2 );
				}
				switch( m_nClassType )
				{
				case Adaboost:	t_nRes = (( CvBoost * )m_pClassifier)->predict( t_FeatureMat );break;
				case Rtree:		t_nRes = (( CvRTrees * )m_pClassifier)->predict( t_FeatureMat );break;
				case SVMC:		t_nRes = (( CvSVM * )m_pClassifier)->predict( t_FeatureMat );  break;
				}	
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
			//测试子图输出模块
			if ( m_bSavePosPatch )
			{	
				if( t_nRes >= 0.5f )
				{
					string t_sSave;
					char p[32] = { 0, };	//初始化临时字符串
					sprintf( p,"%d", g_nPosImageNumber );
					t_sSave = p;
					t_sSave = m_sPosPatchPath + t_sSave;
					g_nPosImageNumber++;
					Mat t_SaveImage;
					t_SaveImage = m_TestImage( t_Rect ).clone();
					ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
					memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );
					t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
				}
			}
			if ( m_bSaveNegPatch )
			{
				if( t_nRes <= 0.5f )
				{
					string t_sSave;
					char p[32] = { 0, };	//初始化临时字符串
					sprintf( p,"%d", g_nNegImageNumber );
					t_sSave = p;
					t_sSave = m_sNegPatchPath + t_sSave;
					g_nNegImageNumber++;
					Mat t_SaveImage;
					t_SaveImage = m_TestImage( t_Rect ).clone();
					ListImage t_Img( t_SaveImage.cols, t_SaveImage.rows, t_SaveImage.channels() );
					memcpy( t_Img.GetImgBuffer(), t_SaveImage.data, t_Img.GetImgDataSize() );
					t_Img.SaveImageToFile( t_sSave.c_str(), LIF_JPEG );
				}
			}
		
			t_Rect.y += m_nSearchStep;
		}
		t_Rect.x += m_nSearchStep;
	}
		delete [] t_pFeature;
	cvReleaseMat( &t_FeatureMat );
	
	return 1;	
	
}//SearchTargetPerImg
/*****************************************************************
Name:			CountFeatureFromImg
Inputs:
	Mat t_Image - 待计算特征的图像
	float *t_pFeatures - 返回的特征数组
Return Value:
	int - 1 正常返回
		  <0 对应错误代码
Description:	计算单张图片的特征
*****************************************************************/
int RHOG::CountFeatureFromImg( Mat t_Image, float *t_pFeatures )
{
	//清空现有数据
	Clear();	
	//初始化特征空间
	InitFeatures();			//开辟特征空间
	//初始化图像
	Mat t_GrayImage;
	PreProcessImage( t_Image, t_GrayImage );
	m_Image.create( m_nImageWidth, m_nImageWidth, CV_32SC1 );
	resize( t_GrayImage, m_Image, Size( m_nImageWidth, m_nImageWidth ), 0, 0, CV_INTER_LINEAR );
	//计算梯度
	CountGrad( m_Image );
	//计算cell特征向量
	CountCell( 0, 0, m_nImageWidth, m_nImageWidth );
	//平滑Cell特征向量
	SmoothCell();
	//对Cell进行角度间的平滑
	if ( m_bRSC )
	{
		RSmoothCell();
	}
	//对Cell进行径向的平滑
	if ( m_bDSC )
	{
		DSmoothCell();
	}
	//计算m_nBIN特征向量
	Countm_nBIN();
	//归一化m_nBIN特征向量
	Normalm_nBIN();
	//计算对称特征
	if ( m_bSym )
	{
		CountSym();
	}
	memcpy( t_pFeatures, m_pfFeature, m_nFeatureNumber * sizeof( float ) );
	return 1;
}//CountFeatureFromImg
/*****************************************************************
Name:			CountFeature
Inputs:
	int t_nX - 提取区域坐标
	int t_nY 
	int t_nWidth
	int t_nHeight
	float *&t_pFeatures - 返回的特征向量
Return Value:
	int - 1 正常返回
		  <0 对应错误代码
Description:	计算指定坐标区域的特征（大图中的局部子块）
*****************************************************************/
int RHOG::CountFeature( int t_nX, int t_nY, int t_nWidth, int t_nHeight, float *t_pFeatures )
{
	//clock_t start1,end1,start2,end2,start3,end3,start4,end4;
	CString out;
//	 cudaEvent_t start,stop,start1,stop1,start2,stop2,start3,stop3;
	 float time_elapsed=0;
//	cudaEventCreate(&start);
//cudaEventCreate(&stop);
//cudaEventRecord(start, 0);
//	
	//_LARGE_INTEGER time_start,s1,s2,s3;  //开始时间  
	//_LARGE_INTEGER time_over,e1,e2,e3;   //结束时间  
	//double dqFreq;      //计时器频率  
	//LARGE_INTEGER f;    //计时器频率  
	//QueryPerformanceFrequency(&f);  
	//dqFreq=(double)f.QuadPart;  
	//QueryPerformanceCounter(&time_start);
	//计算cell特征向量
	CountCell( t_nX, t_nY, t_nWidth, t_nHeight );
	// QueryPerformanceCounter(&time_over);    //计时结束  
	// time_elapsed=1000000*(time_over.QuadPart-time_start.QuadPart)/dqFreq;  
	////乘以1000000把单位由秒化为微秒，精度为1000 000/（cpu主频）微秒  
	//out.Format("Countcell:  %lf us",time_elapsed);
			//AfxMessageBox(out);
			 //QueryPerformanceCounter(&s1);
	//平滑Cell特征向量
	SmoothCell();
	 //QueryPerformanceCounter(&e1);    //计时结束  
	 //time_elapsed=1000000*(e1.QuadPart-s1.QuadPart)/dqFreq;  
	 //out.Format("smoothcell:  %lf us",time_elapsed);
			//AfxMessageBox(out);
	//对Cell进行角度间的平滑
	if ( m_bRSC )
	{
		RSmoothCell();
	}
	//对Cell进行径向的平滑
	if ( m_bDSC )
	{
		DSmoothCell();
	}
	 //QueryPerformanceCounter(&s2);
	
	//计算m_nBIN特征向量
	Countm_nBIN();
	
	 //QueryPerformanceCounter(&e2);    //计时结束  
	 //time_elapsed=1000000*(e2.QuadPart-s2.QuadPart)/dqFreq;  
	 //out.Format("Countm_nBIN():  %lf us",time_elapsed);
			//AfxMessageBox(out);
			//QueryPerformanceCounter(&s3);
	//归一化m_nBIN特征向量
	Normalm_nBIN();
	 //QueryPerformanceCounter(&e3);    //计时结束  
	 //time_elapsed=1000000*(e3.QuadPart-s3.QuadPart)/dqFreq;  
	 //out.Format("Normalm_nBIN():  %lf us",time_elapsed);
			//AfxMessageBox(out);
	//计算对称特征
	if ( m_bSym )
	{
		CountSym();
	}
	//复制待返回的特征值
	memcpy( t_pFeatures, m_pfFeature, m_nFeatureNumber * sizeof( float ) );
	return 1;
}//CountFeature
/*****************************************************************s
Name:			Clear
Inputs:
	none.
Return Value:
	none.
Description:	注销空间
*****************************************************************/
void RHOG::Clear(void)
{
	//清空特征数据
	if ( m_pfFeature != NULL )
	{
		delete [] m_pfFeature;
		m_pfFeature = NULL;
	}
	if ( m_pCellFeatures != NULL )
	{
		delete [] m_pCellFeatures;
		m_pCellFeatures = NULL;
	}
}//Clear
/*****************************************************************
Name:			InitFeatures
Inputs:
	none.
Return Value:
	none.
Description:	初始化特征空间
*****************************************************************/
void RHOG::InitFeatures(void)
{
	//计算
	//清空特征数据
	if ( m_pfFeature != NULL )
	{
		delete [] m_pfFeature;
		m_pfFeature = NULL;
	}
	if ( m_pCellFeatures != NULL )
	{
		delete [] m_pCellFeatures;
		m_pCellFeatures = NULL;
	}
	m_pfFeature = new float [m_nFeatureNumber];					//开辟空间
	m_pCellFeatures = new float [m_nANG * m_nCellNumb * m_nBIN];//开辟空间
	m_nCellWidth = ( m_nImageWidth / 2 ) / m_nCellNumb + 1;		//每个cell的宽度
}//InitFeatures
/*****************************************************************
Name:			PreProcessImage
Inputs:
	Mat t_Image - 输入图像
	 Mat &t_TarImage - 预处理后图像
Return Value:
	none.
Description:	预处理图像
*****************************************************************/
void RHOG::PreProcessImage( Mat t_Image, Mat &t_TarImage )
{
	//转灰度图像，并缩放至统一大小
	Mat t_GrayImage;
	if ( t_Image.channels() == 3 )
	{
		cvtColor( t_Image, t_GrayImage, CV_BGR2GRAY );
	}
	else
	{
		t_GrayImage = t_Image.clone();
	}
	clock_t start1,end1,start2,end2,start3,end3,start4,end4;
	CString out;
	/*unsigned char *src=t_GrayImage.data;
	unsigned char *dst=t_GrayImage.data;*/
	int width;
	width=t_GrayImage.cols;
	int height;
	height=t_GrayImage.rows;
	//滤波
		start4=clock();
	medianBlur( t_GrayImage, t_GrayImage, m_nFilterSize );
	end4=clock();
	out.Format("中值滤波:  %lf ms",double(end4-start4));
			//AfxMessageBox(out);
		start1=clock();
		//MedianFilter(t_GrayImage.data,t_GrayImage.data ,width,height);
		end1=clock();
	out.Format("中值滤波:  %lf ms",double(end1-start1));
			//AfxMessageBox(out);
	//imshow("meidan",t_GrayImage);
	//waitKey(0);
	GaussianBlur( t_GrayImage, t_GrayImage, cvSize( 3, 3 ), 1 );
	//将图像转为float型，并对图像进行归一化处理
			start2=clock();
	t_GrayImage.convertTo( t_TarImage, CV_32F );
	end2=clock();
	out.Format("转灰度图:  %lf ms",double(end2-start2));
			//AfxMessageBox(out);
	////灰度gamma处理
	//暂时去掉
			start3=clock();
	int i, j;
	for ( j = 0; j < t_Image.rows; ++j )
	{
		float *t_pData; 
		t_pData = t_TarImage.ptr<float>(j);
		for ( i = 0; i < t_Image.cols; ++i )
		{
			t_pData[i] = sqrt( t_pData[i] );
		}
	}
	end3=clock();
	out.Format("gamma归一:  %lf ms",double(end3-start3));
			//AfxMessageBox(out);
}//PreProcessImage
/*****************************************************************
Name:			CountGrad
Inputs:
	Mat t_Image - 输入图像
Return Value:
	none.
Description:	计算梯度，并存储至m_MagImage、m_ANGImage图像中
*****************************************************************/
void RHOG::CountGrad( Mat t_Image )
{
	//初始化空间
	m_MagImage.create( t_Image.rows, t_Image.cols, CV_32FC1 );
	m_ANGImage.create( t_Image.rows, t_Image.cols, CV_32FC1 );
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
		t_pPosMag = m_MagImage.ptr<float>(j);
		float *t_pPosm_nANG;		//梯度角度指针
		t_pPosm_nANG = m_ANGImage.ptr<float>(j);
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
/*****************************************************************
Name:			CountCell
Inputs:
	int t_nX - 区域坐标
	int t_nY
	int t_nWidth
	int t_nHeight
Return Value:
	none.
Description:	计算Cell
*****************************************************************/
void RHOG::CountCell( int t_nX, int t_nY, int t_nWidth, int t_nHeight )
{
	int t_nEndX;
	int t_nEndY;
	t_nEndX = t_nX + t_nWidth -1;
	t_nEndY = t_nY + t_nHeight -1;
	int t_nCellFeatureSize;
	t_nCellFeatureSize = m_nANG * m_nCellNumb * m_nBIN;
	int i, j;
	for ( i = 0; i < t_nCellFeatureSize; ++i )
	{
		m_pCellFeatures[i] = 0;
	}
	//生成cellfeature
	int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = m_nCellNumb * m_nBIN;
	for ( j = t_nY + 1; j < t_nEndY; ++j )
	{
		float *t_pMagData;
		float *t_pm_nANGData;
		t_pMagData = m_MagImage.ptr<float>(j);
		t_pm_nANGData = m_ANGImage.ptr<float>(j);
		for ( i = t_nX + 1; i < t_nEndX; ++i )
		{
			//判断是否超出半径
			if ( m_fMagMat[( j - t_nY) * m_nImageWidth + i - t_nX] > m_nImageWidth / 2.0f )
			{
				continue;
			}
			//计算m_nBIN
			float t_fm_nANGel;
			t_fm_nANGel = t_pm_nANGData[i] - m_fNormalMat[( j - t_nY) * m_nImageWidth + i - t_nX];
			while ( t_fm_nANGel < 0 )
			{
				t_fm_nANGel += (float)PI;
			}
			int t_nm_nBIN =  (int)( t_fm_nANGel * m_nBIN / PI );
			//计算扇区编号
			//int t_nCir;
			//t_nCir = (int)( m_fMagMat[( j - t_nY) * m_nImageWidth + i - t_nX] / m_nCellWidth);
			//m_pCellFeatures[t_nLineWidth * m_nANGle[( j - t_nY) * m_nImageWidth + i - t_nX] + t_nCir * m_nBIN + t_nm_nBIN ] += t_pMagData[i];
			m_pCellFeatures[t_nLineWidth * m_nANGle[( j - t_nY) * m_nImageWidth + i - t_nX] + m_nMag[( j - t_nY) * m_nImageWidth + i - t_nX] * m_nBIN + t_nm_nBIN ] += t_pMagData[i];
		}
	}
}//CountCell
/*****************************************************************
Name:			SmoothCell
	Inputs:
none.
Return Value:
	none.
Description:	对Cell进行平滑(m_pCellFeatures)
*****************************************************************/
void RHOG::SmoothCell( void )
{
	int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = m_nCellNumb * m_nBIN;//7*10
	int i, j, k;
	float * t_pTemp;		//临时保存
	t_pTemp = new float [m_nBIN];
	for ( k = 0; k < m_nANG; ++k )//18
	{
		for ( j = 0; j < m_nCellNumb; ++j )//7
		{
			for ( i = 0; i< m_nBIN; ++i )//10
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
				m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//SmoothCell
/*****************************************************************
Name:			RSmoothCell
Inputs:
	none.
Return Value:
	none.
Description:	对Cell进行角度平滑(m_pCellFeatures)
*****************************************************************/
void RHOG::RSmoothCell( void )
{
	int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = m_nCellNumb * m_nBIN;
	int i, j, k;
	float * t_pTemp;		//临时保存中间结果
	t_pTemp = new float [m_nANG];
	for ( k = 0; k < m_nCellNumb; ++k )
	{
		for ( j = 0; j < m_nBIN; ++j )
		{
			for ( i = 0; i < m_nANG; ++i )
			{
				int t_nLeft;
				int t_nRight;
				t_nLeft = ( i - 1 + m_nANG ) % m_nANG;
				t_nRight = ( i + 1 ) % m_nANG;
				t_pTemp[i] = m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] * 0.8f 
					+ m_pCellFeatures[t_nLeft * t_nLineWidth + k * m_nBIN + j] * 0.1f 
					+ m_pCellFeatures[t_nRight * t_nLineWidth + k * m_nBIN + j] * 0.1f;
			}
			for ( i = 0; i < m_nANG; ++i )
			{
				m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//RSmoothCell
/*****************************************************************
Name:			DSmoothCell
Inputs:
	none.
Return Value:
	none.
Description:	对Cell进行径向平滑(m_pCellFeatures)
*****************************************************************/
void RHOG::DSmoothCell( void )
{
	int t_nLineWidth;		//每个方向的宽度
	t_nLineWidth = m_nCellNumb * m_nBIN;
	int i, j, k;
	float * t_pTemp;		//临时保存中间结果
	t_pTemp = new float[m_nCellNumb];
	for ( k = 0; k < m_nANG; ++k )
	{
		for ( j = 0; j < m_nBIN; ++j )
		{
			for ( i = 1; i < m_nCellNumb - 1; ++i )
			{
				t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.5f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i + 1] * 0.25f 
					+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i - 1] * 0.25f;
			}
			for ( i = 1; i < m_nCellNumb - 1; ++i )
			{
				m_pCellFeatures[i * t_nLineWidth + k * m_nBIN + j] = t_pTemp[i];
			}
		}
	}
	delete [] t_pTemp;
}//DSmoothCell
/*****************************************************************
Name:			Countm_nBIN
Inputs:
	none.
Return Value:
	none.
Description:	计算m_nBIN
*****************************************************************/
void RHOG::Countm_nBIN( void )
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
/*****************************************************************
Name:			Normalm_nBIN
Inputs:
	none.
Return Value:
	none.
Description:	归一化m_nBIN
*****************************************************************/
void RHOG::Normalm_nBIN( void )
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
}//Normalm_nBIN
/*****************************************************************
Name:			CountSym
Inputs:
	none.
Return Value:
	none.
Description:	计算对称性特征
*****************************************************************/
void RHOG::CountSym( void )
{
	float * t_pPos;		//指向对称特征起始点
	t_pPos = &m_pfFeature[ m_nANG * m_nBlobNumb * m_nCellPerBlob * m_nBIN ];
	int i,j,k;
	for ( i = 0; i < m_nANG / 2; ++i )
	{
		for ( j = 0; j < m_nCellNumb; ++j )
		{
			float t_fSum;		//记录累加
			t_fSum = 0;
			float t_fAddUpA;	//计算均值
			float t_fAddUpB;
			t_fAddUpA = 0;
			t_fAddUpB = 0;
			for ( k = 0; k < m_nBIN; ++k )
			{
				float t_fA;
				float t_fB;
				t_fA = m_pCellFeatures[ i * m_nCellNumb * m_nBIN + j * m_nBIN + k ];
				t_fB = m_pCellFeatures[ ( m_nANG - i - 1 ) * m_nCellNumb * m_nBIN + j * m_nBIN + ( m_nBIN - k ) ];
				t_fSum += t_fA * t_fB;
				t_fAddUpA += t_fA;
				t_fAddUpB += t_fB;
			}
			t_fSum = t_fSum / ( t_fAddUpA * t_fAddUpB + 1 );
			*t_pPos = t_fSum;
			++t_pPos;
		}
	}
}//CountSym
/*****************************************************************
Name:			RefineTargetSeq
Inputs:
	vector <CvRect> t_vTarget - 输入的目标队列
	iRect *& t_pRect - 返回的目标队列
	int t_nMatchTime  - 统计目标时重叠的次数
Return Value:
	int - >0 目标数量
		  <0 对应错误代码
Description:	重新归并目标队列，采用聚类法
*****************************************************************/
int RHOG::RefineTargetSeq( vector <CvRect> t_vTarget, iRect *& t_pRect, int t_nMatchTime )	
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
		CString out;
	 QueryPerformanceCounter(&time_over);    //计时结束  
	float  time_elapsed=1000000*(time_over.QuadPart-time_start.QuadPart)/dqFreq;  
	//乘以1000000把单位由秒化为微秒，精度为1000 000/（cpu主频）微秒  
   
	out.Format("RefineTarget:  %lf us",time_elapsed);
			//AfxMessageBox(out);
	return t_nTarNUmber;
}//RefineTargetSeq
/*****************************************************************
Name:			GetImageList
Inputs:
	string t_sPath - 返回的路径
	vector <string> t_vFileName - 文件名队列
Return Value:
	int - 图像数量.
Description:	获取原始图像列表
*****************************************************************/
int RHOG::GetImageList( string t_sPath, vector <string> &t_vFileName )
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
/*****************************************************************
Name:			cvtList2Mat
Inputs:
	ListImage *SrcImg - 输入图像
	Mat & t_Image - 输出图像
Return Value:
	int 1 - 正常返回
		<0 对应错误代码 
Description:	将listimage图像转为ipl图像
*****************************************************************/
int RHOG::cvtList2Mat( ListImage *SrcImg, Mat & t_Image )
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
