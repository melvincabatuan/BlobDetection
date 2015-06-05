#include "io_github_melvincabatuan_blobdetection_MainActivity.h"

#include <android/bitmap.h>
#include <stdlib.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

 

/*
 * Class:     io_github_melvincabatuan_blobdetection_MainActivity
 * Method:    decode
 * Signature: (Landroid/graphics/Bitmap;[BI)V
 */
JNIEXPORT void JNICALL Java_io_github_melvincabatuan_blobdetection_MainActivity_decode
  (JNIEnv * pEnv, jobject pClass, jobject pTarget, jbyteArray pSource, jint pFilter){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent;

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// 1. Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// 2. cv::Mat for YUV420sp source
    Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);



/***************************************************************************************************/
    /// Native Image Processing HERE...  
   
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

    /// Blob Detect
    if (pFilter == 1){
     
        Mat temp(srcGray.size(), CV_8UC3);

	SimpleBlobDetector::Params params;

	params.minThreshold = 10;
	params.maxThreshold = 200;

	params.filterByArea = false;
	params.filterByCircularity = false;
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;   

        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
 
        std::vector<KeyPoint> keypoints;
	detector->detect( srcGray, keypoints);

        drawKeypoints( srcGray, keypoints, temp, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        cvtColor(temp, mbgra, CV_BGR2BGRA);

    } 

    /// Blob Detect  
    else if (pFilter == 2){

        Mat lowTemp(srcGray.size(), CV_8UC3);
	SimpleBlobDetector::Params lowParams;

	lowParams.minThreshold = 10;
	lowParams.maxThreshold = 200;

	lowParams.filterByArea = true;	
        lowParams.minArea = 100;
	lowParams.filterByCircularity = true;
	lowParams.minCircularity = 0.01;
	lowParams.filterByConvexity = false;
	lowParams.filterByInertia = false;  

        Ptr<SimpleBlobDetector> lowDetector = SimpleBlobDetector::create(lowParams); 
        std::vector<KeyPoint> lowKeypoints;
	lowDetector->detect( srcGray, lowKeypoints);

        drawKeypoints( srcGray, lowKeypoints, lowTemp, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        cvtColor(lowTemp, mbgra, CV_BGR2BGRA);

    }

    /// Gray
    else {
        Mat highTemp(srcGray.size(), CV_8UC3);
	SimpleBlobDetector::Params highParams;

	highParams.minThreshold = 10;
	highParams.maxThreshold = 200;

	highParams.filterByArea = true;	
        highParams.minArea = 500;
	highParams.filterByCircularity = false;
	highParams.filterByConvexity = false;
	highParams.filterByInertia = false;  

        Ptr<SimpleBlobDetector> highDetector = SimpleBlobDetector::create(highParams); 
        std::vector<KeyPoint> highKeypoints;
	highDetector->detect( srcGray, highKeypoints);

        drawKeypoints( srcGray, highKeypoints, highTemp, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        cvtColor(highTemp, mbgra, CV_BGR2BGRA);
    }
/***************************************************************************************************/


    /// Release Java byte buffer and unlock backing bitmap
    pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}
