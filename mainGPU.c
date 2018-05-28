#include "sourcev2.h"
#include <time.h>
using namespace cv;
 extern "C" void countFeaturesfloat(float *img,float SVM_bias,float *svm_weight,int svm_count,
	 int Imagewidth,int ImageHeight,
	float t_fStartRate ,float t_fStepSize,int t_nResizeStep,vector <CvRect> &t_vTarget)
;
void Mat2array( Mat src,float *dst)
{
	int i,j;
	int width=src.cols;
	int height=src.rows;
	
	src.convertTo( src, CV_32F );	
	for ( j = 0; j < height; ++j )
	{
		 float * t_pData= src.ptr<float>(j);
		for ( i = 0; i <width; ++i )
		{
			//t_pData[i] = sqrt( t_pData[i] );
			dst[j*width+i]=t_pData[i];
		}
	}

}
void Mat2arrayS( Mat src,float *dst)
{
	int i,j;
	int width=src.cols;
	int height=src.rows;
	
	src.convertTo( src, CV_32F );	
	for ( j = 0; j < height; ++j )
	{
		 float * t_pData= src.ptr<float>(j);
		for ( i = 0; i <width; ++i )
		{
			t_pData[i] = sqrt( t_pData[i] );
			dst[j*width+i]=t_pData[i];
		}
	}

}
void preprocess(Mat &t_Image )
{
	cvtColor(t_Image,t_Image,CV_BGR2GRAY);
	//imshow("gray",t_Image);
	//waitKey(0);
	medianBlur( t_Image, t_Image, 3);
	//imshow("medianBlur",t_Image);
	//waitKey(0);
	GaussianBlur( t_Image, t_Image, cvSize( 3, 3 ), 1 );
	
}
void getFiles( string path, string savePath,vector<string>& files , vector<string>& savefiles) {  
  
	struct _finddata_t  file;  
   intptr_t  If;  If = _findfirst(path.append("\\*").c_str(), &file);
   //if (() == -1) //不加*也会报错  
   //{  
	  // //std::cout << "Not find image file" << std::endl;  
   //}  
	
   //else{
	   while (_findnext(If, &file) == 0)  
	   {  
		   files.push_back( path.substr(0, path.length() - 1)+file.name );//<< file.name << std::endl; 
		   savefiles.push_back(savePath.substr(0, path.length() - 1)+file.name);
	   }
   //}
	  _findclose(If);  
}  
int cvtList2Ipl( ListImage *SrcImg, IplImage *& t_pImage )
{
	t_pImage = cvCreateImage( cvSize( SrcImg->GetImgWidth(), SrcImg->GetImgHeight() ),  SrcImg->GetImgBPP()/SrcImg->GetImgChannel(), SrcImg->GetImgChannel() );

	int i;
	UCHAR * t_pSrc;
	t_pSrc = SrcImg->GetImgBuffer();
	for ( i = 0; i < t_pImage->imageSize; ++i )
	{
		t_pImage->imageData[i] = t_pSrc[i];
	}

	return 1;
}
int cvtList2Mat( ListImage *SrcImg, Mat & t_Image )
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
int main()
{
	CvSVM  t_pSVM ;
	string t_sClassFilePath="C:\\Users\\Cyj\\Desktop\\trainAtomic.xml";//"C:\\Users\\Cyj\\Desktop\\att321.xml";//"C:\\Users\\Cyj\\Desktop\\atomic323.xml";//"C:\\Users\\Cyj\\Desktop\\trainAtomic.xml";
	t_pSVM.load( t_sClassFilePath.c_str() );
	const int svm_count=t_pSVM.get_var_count();
	float svm_bias=-3.3912737032743312e+000;//-1.9810446822500634e+000;//-2.0699734225461568e+000;//-3.3912737032743312e+000;//-2.7828561096231148e+000;//
	float *svm_weights;

	svm_weights=(float *)malloc(sizeof(float)*svm_count);
	float **svm_all=(float **)malloc(sizeof(float*)*18);//(float*)malloc(sizeof(float)*svm_count*18);
	float *svm_all1=(float*)malloc(sizeof(float)*2160*18);
	for(int j=0;j<18;j++)
		svm_all[j]=(float*)malloc(2160*sizeof(float));
	int  i=0;
 float res=0;
 

 while(i<svm_count)
 {
	 svm_weights[i]=t_pSVM.get_support_vector(0)[i];
	 i++;
 } 
 for(int j=0;j<18;j++)
 {
	 memcpy( svm_all[j],  &svm_weights[j * 120], (18 - j) * 120 * sizeof ( float ) );

				if ( j > 0 )
				{
					memcpy( &svm_all[j][(18 - j) * 120], svm_weights, j * 120* sizeof ( float ) );
				}
	
 }
 for (int j=0;j<18;j++)
	 memcpy(&svm_all1[j*2160],svm_all[j],sizeof(float)*2160);
 /* FILE *fp;
 fp=fopen("G://svm_all1.txt","w+");
 for(int j=0;j<18;j++)
	{ for(int k=0;k<18;k++)
		{ for(int m=k*120;m<120*(k+1);m++)
			 fprintf(fp," %f",svm_all1[m+j*2160]);
 fprintf(fp,"\n");
 }
	fprintf(fp,"\n");		
			
 }

fclose(fp);*/
 
// FILE *fp;
// fp=fopen("G://svm_all.txt","w+");
// for(int j=0;j<18;j++)
//	{ for(int k=0;k<18;k++)
//		{ for(int m=k*120;m<120*(k+1);m++)
//			 fprintf(fp," %f",svm_all[j][m]);
// fprintf(fp,"\n");
// }
//	fprintf(fp,"\n");		
//			
// }
//
//fclose(fp);
// 
 iRect *t_Rect=NULL;

		int num; HOG_CPU cpu_test;

	string m_TestPath="c:\\users\\cyj\\desktop\\atomic2\\D1\\";//"..\\D1\\";//"C:\\Users\\Cyj\\Desktop\\D1\\";//"G:\\DOTA\\JPEGImages\\";//"C:\\Users\\Cyj\\Pictures\\";//"G:\\DOTA\\JPEGImages\\";//"C:\\Users\\Cyj\\Desktop\\D1\\";//"..\\D1\\";//;//"C:\\Users\\Cyj\\Desktop\\D1\\";//;//"C:\\Users\\Cyj\\DesktopD1\\";//"C:\\Users\\Cyj\\Desktop\\Atomic2\\D1\\";//;//;//;//"G:\\DOTA\\images";///;
	string m_SavePath="C:\\Users\\Cyj\\Desktop\\save\\";
	vector<string> imageList;//用于保存图像名称队列
	vector<string> imageSaveList;//用于保存  要保存的各副图片的路径及名称
	vector<string> files;  
	getFiles(m_TestPath, m_SavePath,files ,imageSaveList);  //files为返回的文件名构成的字符串向量组  
		float elapsed=0;
	
	 for (i=1;i<files.size();i++)
	{
		 clock_t start,stop,start1,stop1;
		num=0;
		vector <CvRect> t_vTarget;
		string c_name(files[i]);
		//c_name=imageList[i].GetBuffer();
		ListImage tmpListImage;
		tmpListImage.LoadImageFromFile(c_name.c_str());
		//
		Mat src;
		////IplImage *img=cvLoadImage(imageList[i]);
	
	 //
		//
		cvtList2Mat(&tmpListImage,src); 
		//src=imread("C:\\Users\\Cyj\\Desktop\\Atomic2\\P0000.jpg",1);
		//src=imread("C:\\Users\\Cyj\\Desktop\\Atomic2\\P0000.jpg");//("C:\\Users\\Cyj\\Desktop\\Atomic2\\D1\\test.jpg");
		Mat dst_Image(src.cols,src.rows,CV_8UC1);
		start=clock(); 
		cpu_test.preprocess(src,dst_Image);
		stop=clock(); 
		int width=src.cols;
		int height =src.rows;
		float *srcIm=(float *)malloc(sizeof(float)*width*height);
		Mat2array(dst_Image,srcIm);		
	start1=clock();
		countFeaturesfloat(srcIm,svm_bias,svm_all1,svm_count,width,height,0.33f,0.1f,3,t_vTarget);
		num=cpu_test.RefineTargetSeq(t_vTarget,t_Rect,5);
			stop1=clock();
			 if(i>1)
			 elapsed+=stop1-start1+(stop-start);
			 //if(i>2)break;//
		//cout<<"第"<<i<<"张 "<<"width*height:"<<width<<"x"<<height<<" \n gpu test time:"<<(stop-start)<<"ms"<<endl;
			
		std::cout<<"第"<<i<<"张 "<<"width*height:"<<width<<"x"<<height<<" \n Gpu test time:"<<(stop1-start1)<<"ms"<<endl;
			
		//printf("GPU test time:%d ms\n",stop-start);
		IplImage *t_Image=NULL;
		 cvtList2Ipl(&tmpListImage,t_Image);
		//t_Image=cvLoadImage("C:\\Users\\Cyj\\Desktop\\Atomic2\\P0000.jpg");
		//t_Image=cvLoadImage("..\\D1\\test.jpg");
	/*	for (int j=0;j<num;j++)
		{
			cvRectangle(t_Image,cvPoint(t_Rect[j].x,t_Rect[j].y),cvPoint(t_Rect[j].x+t_Rect[j].m_nWidth,t_Rect[j].y+t_Rect[j].m_nHeight),cvScalar(0xff,0x00,0x00),2);
		}
		namedWindow("test",0);
		cvShowImage("test",t_Image);
		waitKey();*/
/*	t_vTarget.clear();*/		//cvSaveImage(imageSaveList[i].c_str(),t_Image);
			t_vTarget.clear();
		if (t_Rect!=NULL)
	{
		delete[] t_Rect;
		t_Rect=NULL;
	}
		cvReleaseImage(&t_Image);
		
	 }
	  cout<<"总共"<<i<<"张 "<<" \n gpu 总时间:"<<elapsed<<"s"<<endl;
	  getchar();
		/*for (j=0;j<num;j++)
		{
			cvRectangle(t_Image,cvPoint(t_Rect[j].x,t_Rect[j].y),cvPoint(t_Rect[j].x+t_Rect[j].m_nWidth,t_Rect[j].y+t_Rect[j].m_nHeight),cvScalar(0xff,0x00,0x00),2);
		
		}
 cvShowImage("IplImage",t_Image); 
		 waitKey();*/
		
	return 0;
}