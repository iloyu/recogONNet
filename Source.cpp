#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
#define windowx 128
extern "C" void countFeaturesfloat(uchar *in,float *out,int *d_ANG,int *d_Mag,float *c_ANG,float *c_Mag,float *p_ANG,float *p_Mag,int Imagewidth,int ImageHeight);

int main()
{
	Mat src=imread("C:\\Users\\Cyj\\Desktop\\test.jpg",CV_LOAD_IMAGE_UNCHANGED);
	cvtColor(src,src,CV_BGR2GRAY);
	float *dst,*m_fNormalMat,*m_fMagMat,*m_ANGImage,*m_MagImage;
	int *m_nANG,*m_nMag;
	int width=src.cols;
	int height=src.rows;
	long sizem_p=sizeof(float)*width*height;
	long sizem_n=sizeof(int)*128*128;
	long sizem_f=sizeof(float)*128*128;
	m_nANG=(int *)malloc(sizem_n);
	m_nMag=(int *)malloc(sizem_n);
	m_fMagMat=(float *)malloc(sizem_f);
	m_fNormalMat=(float*)malloc(sizem_f);
	m_MagImage=(float*)malloc(sizem_p);
	m_ANGImage=(float*)malloc(sizem_p);


	int h_windowx=src.cols/128;
	int h_windowy=src.rows/128;
	dst=(float*)malloc(sizeof(float)*h_windowx*h_windowy*2160);
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
  fp=fopen("G://m_nfNormalMat.txt","r");
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		fscanf(fp," %f ",&m_fNormalMat[i*windowx+j]);
 }
 fclose(fp);
  fp=fopen("G://m_nfMagMat.txt","r");long count=0;
 for (int i = 0; i <windowx ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<windowx;j++)
		{fscanf(fp," %f ",&m_fMagMat[i*windowx+j]);
	
	}

 }
 //printf(" %f ",count);
 fclose(fp);
 fp=fopen("G://m_ANGLEPixel.txt","r");
 for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
		fscanf(fp," %f ",&m_ANGImage [i*width+j]);
 }
 fclose(fp);
 fp=fopen("G://m_MagPixel.txt","r");
 for (int i = 0; i <height ; i++) {  //从fp指向的文件中读取10个整数到b数组
    for(int j=0;j<width;j++)
		fscanf(fp," %f ",&m_MagImage[i*width+j]);
 }
 fclose(fp);

 countFeaturesfloat(src.data,dst,m_nANG,m_nMag,m_fNormalMat,m_fMagMat,m_ANGImage,m_MagImage,width,height);
	fp=fopen("G://out.txt","w");
	for (int i = 0; i < h_windowy; i++) {  //从fp指向的文件中读取10个整数到b数组
		for(int j=0;j<h_windowx;j++)
			for (int k=0;k<2160;k++)
				fprintf(fp," %f ",&dst[((i*h_windowx)+j)*2160+k]);
 }
 fclose(fp);
 return 0;
}