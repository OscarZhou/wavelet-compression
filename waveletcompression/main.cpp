#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <set>
#include <utility>

using namespace std;
using namespace cv;

#define THRESHOLD 1


float threshold_filter(float input, int threshold) {
    if (input < threshold) return 0;
    else return input;
}


#if (THRESHOLD == 1)


// 二维离散小波变换（单通道浮点图像）
void DWT(IplImage * pImage, int nLayer, int threshold)
#else
void DWT(IplImage * pImage, int nLayer)
# endif 
{
    // 执行条件
    if (pImage) {
        if (pImage->nChannels == 1 &&
            pImage->depth == IPL_DEPTH_32F &&
            ((pImage->width >> nLayer) << nLayer) == pImage->width &&
            ((pImage->height >> nLayer) << nLayer) == pImage->height) {
            int i, x, y, n;
            float fValue = 0;
            float fRadius = sqrt(2.0f);
            int nWidth = pImage->width;
            int nHeight = pImage->height;
            int nHalfW = nWidth / 2;
            int nHalfH = nHeight / 2;
            float * * pData = new float * [pImage->height];
            float * pRow = new float[pImage->width];
            float * pColumn = new float[pImage->height];
            for (i = 0; i < pImage->height; i++) {
                pData[i] = (float * )(pImage->imageData + pImage->widthStep * i);
            }
            // 多层小波变换
            for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2) {
                // 水平变换
                for (y = 0; y < nHeight; y++) {
                    // 奇偶分离
                    memcpy(pRow, pData[y], sizeof(float) * nWidth);
                    for (i = 0; i < nHalfW; i++) {
                        x = i * 2;
                        pData[y][i] = pRow[x];
                        pData[y][nHalfW + i] = pRow[x + 1];
                    }
                    // 提升小波变换
                    for (i = 0; i < nHalfW - 1; i++) {
                        fValue = (pData[y][i] + pData[y][i + 1]) / 2;
                        pData[y][nHalfW + i] -= fValue;
                    }
                    fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
                    pData[y][nWidth - 1] -= fValue;
                    fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
                    pData[y][0] += fValue;
                    for (i = 1; i < nHalfW; i++) {
                        fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
                        pData[y][i] += fValue;
                    }
                    // 频带系数
                    for (i = 0; i < nHalfW; i++) {
                        pData[y][i] *= fRadius;
                        pData[y][nHalfW + i] /= fRadius;
#if (THRESHOLD == 1)
                            pData[y][i] = threshold_filter(pData[y][i], threshold);
                        pData[y][nHalfW + i] = threshold_filter(pData[y][nHalfW + i], threshold);
#endif
                    }
                }
                // 垂直变换
                for (x = 0; x < nWidth; x++) {
                    // 奇偶分离
                    for (i = 0; i < nHalfH; i++) {
                        y = i * 2;
                        pColumn[i] = pData[y][x];
                        pColumn[nHalfH + i] = pData[y + 1][x];
                    }
                    for (i = 0; i < nHeight; i++) {
                        pData[i][x] = pColumn[i];
                    }
                    // 提升小波变换
                    for (i = 0; i < nHalfH - 1; i++) {
                        fValue = (pData[i][x] + pData[i + 1][x]) / 2;
                        pData[nHalfH + i][x] -= fValue;
                    }
                    fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
                    pData[nHeight - 1][x] -= fValue;
                    fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
                    pData[0][x] += fValue;
                    for (i = 1; i < nHalfH; i++) {
                        fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
                        pData[i][x] += fValue;
                    }
                    // 频带系数
                    for (i = 0; i < nHalfH; i++) {
                        pData[i][x] *= fRadius;
                        pData[nHalfH + i][x] /= fRadius;
#if (THRESHOLD == 1)
                            pData[i][x] = threshold_filter(pData[i][x], threshold);
                        pData[nHalfH + i][x] = threshold_filter(pData[nHalfH + i][x], threshold);
#endif
                    }
                }
            }
            delete[] pData;
            delete[] pRow;
            delete[] pColumn;
        }
    }
}

// 二维离散小波恢复（单通道浮点图像）
void IDWT(IplImage * pImage, int nLayer) {
    // 执行条件
    if (pImage) {
        if (pImage->nChannels == 1 &&
            pImage->depth == IPL_DEPTH_32F &&
            ((pImage->width >> nLayer) << nLayer) == pImage->width &&
            ((pImage->height >> nLayer) << nLayer) == pImage->height) {
            int i, x, y, n;
            float fValue = 0;
            float fRadius = sqrt(2.0f);
            int nWidth = pImage->width >> (nLayer - 1);
            int nHeight = pImage->height >> (nLayer - 1);
            int nHalfW = nWidth / 2;
            int nHalfH = nHeight / 2;
            float * * pData = new float * [pImage->height];
            float * pRow = new float[pImage->width];
            float * pColumn = new float[pImage->height];
            for (i = 0; i < pImage->height; i++) {
                pData[i] = (float * )(pImage->imageData + pImage->widthStep * i);
            }
            // 多层小波恢复
            for (n = 0; n < nLayer; n++, nWidth *= 2, nHeight *= 2, nHalfW *= 2, nHalfH *= 2) {
                // 垂直恢复
                for (x = 0; x < nWidth; x++) {
                    // 频带系数
                    for (i = 0; i < nHalfH; i++) {
                        pData[i][x] /= fRadius;
                        pData[nHalfH + i][x] *= fRadius;
                    }
                    // 提升小波恢复
                    fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
                    pData[0][x] -= fValue;
                    for (i = 1; i < nHalfH; i++) {
                        fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
                        pData[i][x] -= fValue;
                    }
                    for (i = 0; i < nHalfH - 1; i++) {
                        fValue = (pData[i][x] + pData[i + 1][x]) / 2;
                        pData[nHalfH + i][x] += fValue;
                    }
                    fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
                    pData[nHeight - 1][x] += fValue;
                    // 奇偶合并
                    for (i = 0; i < nHalfH; i++) {
                        y = i * 2;
                        pColumn[y] = pData[i][x];
                        pColumn[y + 1] = pData[nHalfH + i][x];
                    }
                    for (i = 0; i < nHeight; i++) {
                        pData[i][x] = pColumn[i];
                    }
                }
                // 水平恢复
                for (y = 0; y < nHeight; y++) {
                    // 频带系数
                    for (i = 0; i < nHalfW; i++) {
                        pData[y][i] /= fRadius;
                        pData[y][nHalfW + i] *= fRadius;
                    }
                    // 提升小波恢复
                    fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
                    pData[y][0] -= fValue;
                    for (i = 1; i < nHalfW; i++) {
                        fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
                        pData[y][i] -= fValue;
                    }
                    for (i = 0; i < nHalfW - 1; i++) {
                        fValue = (pData[y][i] + pData[y][i + 1]) / 2;
                        pData[y][nHalfW + i] += fValue;
                    }
                    fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
                    pData[y][nWidth - 1] += fValue;
                    // 奇偶合并
                    for (i = 0; i < nHalfW; i++) {
                        x = i * 2;
                        pRow[x] = pData[y][i];
                        pRow[x + 1] = pData[y][nHalfW + i];
                    }
                    memcpy(pData[y], pRow, sizeof(float) * nWidth);
                }
            }
            delete[] pData;
            delete[] pRow;
            delete[] pColumn;
        }
    }
}

#if (THRESHOLD != 1)

int main(int argc, char * * argv) {
    // 小波变换层数
    int nLayer = 1;
    // 输入彩色图像
    IplImage * pSrc = cvLoadImage(argv[1], 1);
    // 计算小波图象大小，使其width和height都是2的倍数
    CvSize size = cvGetSize(pSrc);
    CvSize size1 = cvGetSize(pSrc);
    if ((pSrc->width >> nLayer) << nLayer != pSrc->width) {
        size.width = ((pSrc->width >> nLayer) + 1) << nLayer;

    }
    if ((pSrc->height >> nLayer) << nLayer != pSrc->height) {
        size.height = ((pSrc->height >> nLayer) + 1) << nLayer;

    }
    // 创建小波图象
    IplImage * pWavelet = cvCreateImage(size, IPL_DEPTH_32F, pSrc->nChannels);
    size1.width = pSrc->width >> nLayer;
    size1.height = pSrc->height >> nLayer;
    cout << "size = " << size.width << ", " << size.height << endl;
    cout << "size = " << size1.width << ", " << size1.height << endl;
    IplImage * pWavelet1 = cvCreateImage(size1, IPL_DEPTH_32F, pSrc->nChannels);
    if (pWavelet) {
        // 小波图象赋值
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pSrc, pWavelet, 1, -128); //使用线性变换转换数组：pWavelet = pSrc*1-128
        cvResetImageROI(pWavelet);
        // 彩色图像小波变换
        IplImage * pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);
        if (pImage) {
            for (int i = 1; i <= pWavelet->nChannels; i++) {
                cvSetImageCOI(pWavelet, i); //设置感兴趣通道channel
                cvCopy(pWavelet, pImage, NULL); //pImage为灰度图像，将pWavelet的每个通道数据copy到pImage中
                // 二维离散小波变换
                DWT(pImage, nLayer); //对每个通道进行DWT
                // 二维离散小波恢复
                IDWT(pImage, nLayer);
                cvCopy(pImage, pWavelet, NULL); //将每个通道变换后的数据存入pWavelet的对应通道中
            }
            cvSetImageCOI(pWavelet, 0);
            cvReleaseImage( &pImage);
        }
        // 小波变换图象
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pWavelet, pSrc, 1, 128);
        cvResetImageROI(pWavelet); // 本行代码有点多余，但有利用养成良好的编程习惯
        cvReleaseImage( &pWavelet);
    }
    // 显示图像pSrc

/*
    cvSetImageROI(pSrc, cvRect(0, 0, size1.width, size1.height)); //设置源图像ROI
    IplImage * pDest = cvCreateImage(size1, pSrc->depth, pSrc->nChannels); //创建目标图像
    cvCopy(pSrc, pDest); //复制图像
    cvResetImageROI(pDest); //源图像用完后，清空ROI
    cvSaveImage("lenna_haar.jpg", pDest); //保存目标图像

*/
    cvSetImageROI(pSrc, cvRect(0, 0, size.width, size.height)); //设置源图像ROI
    IplImage * pDest = cvCreateImage(size, pSrc->depth, pSrc->nChannels); //创建目标图像
    cvCopy(pSrc, pDest); //复制图像
    cvResetImageROI(pDest); //源图像用完后，清空ROI
    cvSaveImage("lenna_haar.jpg", pDest); //保存目标图像


    cvNamedWindow("dwt", 1);
    cvShowImage("dwt", pSrc);
    cvWaitKey(0);
    cvDestroyWindow("dwt");
    // ...
    cvReleaseImage( &pSrc);
    //cvReleaseImage(&pSrc0);

    return 0;
    return 0;
}

#else

int main(int argc, char * * argv) {
    // 小波变换层数
    int nLayer = 1;
    // 输入彩色图像
    IplImage * pSrc = cvLoadImage(argv[1], 1);
    // 计算小波图象大小，使其width和height都是2的倍数
    CvSize size = cvGetSize(pSrc);
    CvSize size1 = cvGetSize(pSrc);
    if ((pSrc->width >> nLayer) << nLayer != pSrc->width) {
        size.width = ((pSrc->width >> nLayer) + 1) << nLayer;

    }
    if ((pSrc->height >> nLayer) << nLayer != pSrc->height) {
        size.height = ((pSrc->height >> nLayer) + 1) << nLayer;

    }
    // 创建小波图象
    IplImage * pWavelet = cvCreateImage(size, IPL_DEPTH_32F, pSrc->nChannels);
    size1.width = pSrc->width >> nLayer;
    size1.height = pSrc->height >> nLayer;
    cout << "size = " << size.width << ", " << size.height << endl;
    cout << "size = " << size1.width << ", " << size1.height << endl;
    IplImage * pWavelet1 = cvCreateImage(size1, IPL_DEPTH_32F, pSrc->nChannels);

    if (pWavelet) {
        // 小波图象赋值
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        //vSetImageROI(pWavelet1, cvRect(0, 0, pWavelet1->width, pWavelet1->height));

        cvConvertScale(pSrc, pWavelet, 1, -128); //使用线性变换转换数组：pWavelet = pSrc*1-128

        cvResetImageROI(pWavelet1);
        // 彩色图像小波变换
        IplImage * pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);

        if (pImage) {
            for (int i = 1; i <= pWavelet->nChannels; i++) {
                cvSetImageCOI(pWavelet, i); //设置感兴趣通道channel
                cvCopy(pWavelet, pImage, NULL); //pImage为灰度图像，将pWavelet的每个通道数据copy到pImage中
                // 二维离散小波变换
                DWT(pImage, nLayer, stoi(argv[2])); //对每个通道进行DWT
                // 二维离散小波恢复
                IDWT(pImage, nLayer);
                cvCopy(pImage, pWavelet, NULL); //将每个通道变换后的数据存入pWavelet的对应通道中
            }
            cvSetImageCOI(pWavelet, 0);
            cvReleaseImage( &pImage);

            //cout<<"111"<<endl;
        }

        // 小波变换图象
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        //cvSetImageROI(pWavelet1, cvRect(0, 0, pWavelet1->width, pWavelet1->height));
        cvConvertScale(pWavelet, pSrc, 1, 128);
        cvResetImageROI(pWavelet); // 本行代码有点多余，但有利用养成良好的编程习惯
        cvReleaseImage( &pWavelet);
    }
    // 显示图像pSrc
    cvNamedWindow("dwt", 1);
    cvShowImage("dwt", pSrc);

    //CvSize size= cvSize(size1.width,size1.height);//区域大小

    cvSetImageROI(pSrc, cvRect(0, 0, size.width, size.height)); //设置源图像ROI
    IplImage * pDest = cvCreateImage(size, pSrc->depth, pSrc->nChannels); //创建目标图像
    cvCopy(pSrc, pDest); //复制图像
    cvResetImageROI(pDest); //源图像用完后，清空ROI
    cvSaveImage("lenna_haar_50.png", pDest); //保存目标图像

    //cvSaveImage("lenahaar.jpg", pWavelet1);

    cvWaitKey(0);
    cvDestroyWindow("dwt");

    // ...
    cvReleaseImage( &pSrc);
    //cvReleaseImage(&pSrc0);

    return 0;
    return 0;
}
#endif
