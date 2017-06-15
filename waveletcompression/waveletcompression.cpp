#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <set>
#include <utility>

using namespace std;
using namespace cv;

#define THRESHOLD_VERSION 0


// written by Hongyu ZHOU(Oscar) , 16242950 and Bohong CAO(Echo), 15306483

/*********************************************************************************************
 * Compile with:
 * g++ -std=c++0x -o waveletcompression -O3 waveletcompression.cpp `pkg-config --libs --cflags opencv`
 * Execute static code: for example
 * ./waveletcompression ~/Downloads/lena.jpg
 *
*********************************************************************************************/




float threshold_filter(float input, int threshold) {
    if (input < threshold) return 0;
    else return input;
}


#if (THRESHOLD_VERSION == 1)


// 2-D Continuous Wavelet Transform（single channel floating-point images）
void DWT(IplImage * pImage, int nLayer, int threshold)
#else
void DWT(IplImage * pImage, int nLayer)
# endif 
{
    // Execution Condition
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
            // multilayer wavelet transformation
            for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2) {
                // horizontal transformation
                for (y = 0; y < nHeight; y++) {
                    // Parity separation
                    memcpy(pRow, pData[y], sizeof(float) * nWidth);
                    for (i = 0; i < nHalfW; i++) {
                        x = i * 2;
                        pData[y][i] = pRow[x];
                        pData[y][nHalfW + i] = pRow[x + 1];
                    }
                    // improve wavelet transformation
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
                    // band factor
                    for (i = 0; i < nHalfW; i++) {
                        pData[y][i] *= fRadius;
                        pData[y][nHalfW + i] /= fRadius;
#if (THRESHOLD_VERSION == 1)
                        pData[y][i] = threshold_filter(pData[y][i], threshold);
                        pData[y][nHalfW + i] = threshold_filter(pData[y][nHalfW + i], threshold);
#endif
                    }
                }
                // vertical transformation
                for (x = 0; x < nWidth; x++) {
                    // Parity separation
                    for (i = 0; i < nHalfH; i++) {
                        y = i * 2;
                        pColumn[i] = pData[y][x];
                        pColumn[nHalfH + i] = pData[y + 1][x];
                    }
                    for (i = 0; i < nHeight; i++) {
                        pData[i][x] = pColumn[i];
                    }
                    // improve wavelet transformation
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
                    // band factor
                    for (i = 0; i < nHalfH; i++) {
                        pData[i][x] *= fRadius;
                        pData[nHalfH + i][x] /= fRadius;
#if (THRESHOLD_VERSION == 1)
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

// two-dimension discrete wavelet recover（single channel floating-point images）
void IDWT(IplImage * pImage, int nLayer) {
    // Execution Condition
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
            // multilayer wavelet recover
            for (n = 0; n < nLayer; n++, nWidth *= 2, nHeight *= 2, nHalfW *= 2, nHalfH *= 2) {
                // vertical recover
                for (x = 0; x < nWidth; x++) {
                    // band factor
                    for (i = 0; i < nHalfH; i++) {
                        pData[i][x] /= fRadius;
                        pData[nHalfH + i][x] *= fRadius;
                    }
                    // improve wavelet recover
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
                    // odd-even merging
                    for (i = 0; i < nHalfH; i++) {
                        y = i * 2;
                        pColumn[y] = pData[i][x];
                        pColumn[y + 1] = pData[nHalfH + i][x];
                    }
                    for (i = 0; i < nHeight; i++) {
                        pData[i][x] = pColumn[i];
                    }
                }
                // horizontal recover
                for (y = 0; y < nHeight; y++) {
                    // band factor
                    for (i = 0; i < nHalfW; i++) {
                        pData[y][i] /= fRadius;
                        pData[y][nHalfW + i] *= fRadius;
                    }
                    // improve wavelet recover
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
                    // odd-even merging
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

#if (THRESHOLD_VERSION != 1)

int main(int argc, char * * argv) {
    // wavelet transformation layer
    int nLayer = 2;
    // input color image
    IplImage * pSrc = cvLoadImage(argv[1], 1);
    // calculate the wavelet image so that the width and height are multiples of 2
    CvSize size = cvGetSize(pSrc);
    CvSize size1 = cvGetSize(pSrc);
    if ((pSrc->width >> nLayer) << nLayer != pSrc->width) {
        size.width = ((pSrc->width >> nLayer) + 1) << nLayer;

    }
    if ((pSrc->height >> nLayer) << nLayer != pSrc->height) {
        size.height = ((pSrc->height >> nLayer) + 1) << nLayer;

    }
    // create the wavelet image
    IplImage * pWavelet = cvCreateImage(size, IPL_DEPTH_32F, pSrc->nChannels);
    size1.width = pSrc->width >> nLayer;
    size1.height = pSrc->height >> nLayer;
    cout << "size = " << size.width << ", " << size.height << endl;
    cout << "size = " << size1.width << ", " << size1.height << endl;
    if (pWavelet) {
        // evaluation of wavelet image
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pSrc, pWavelet, 1, -128); //use linear transformation to convert to array：pWavelet = pSrc*1-128
        cvResetImageROI(pWavelet);
        // wavelet transformation of color image
        IplImage * pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);
        if (pImage) {
            for (int i = 1; i <= pWavelet->nChannels; i++) {
                cvSetImageCOI(pWavelet, i); //set interesting channel
                cvCopy(pWavelet, pImage, NULL); //pImage is grayscale image，copy every data of channel in pWavelet to pImage
                // 2-D Continuous Wavelet Transform
                DWT(pImage, nLayer); //DWT every channel
                // 2-D Continuous Wavelet recover
                //IDWT(pImage, nLayer);
                cvCopy(pImage, pWavelet, NULL); // save every data after the transformation of channels into the channel of pWavelet
            }
            cvSetImageCOI(pWavelet, 0);
            cvReleaseImage( &pImage);
        }
        // wavelet transformation image
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pWavelet, pSrc, 1, 128);
        cvResetImageROI(pWavelet); // this line is not necessary but it’s good for the coding habit
        cvReleaseImage( &pWavelet);
    }
    // display image pSrc

    cvSetImageROI(pSrc, cvRect(0, 0, size.width, size.height)); //set source image ROI
    IplImage * pDest = cvCreateImage(size, pSrc->depth, pSrc->nChannels); //set target image
    cvCopy(pSrc, pDest); //copy image
    cvResetImageROI(pDest); //empty ROI after using the source image
    cvSaveImage("lenna_haar.jpg", pDest); //save target image


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
    // layer of wavelet transformation
    int nLayer = 1;
    // input color image
    IplImage * pSrc = cvLoadImage(argv[1], 1);
    // calculate the wavelet image so that the width and height are multiples of 2
    CvSize size = cvGetSize(pSrc);
    CvSize size1 = cvGetSize(pSrc);
    if ((pSrc->width >> nLayer) << nLayer != pSrc->width) {
        size.width = ((pSrc->width >> nLayer) + 1) << nLayer;

    }
    if ((pSrc->height >> nLayer) << nLayer != pSrc->height) {
        size.height = ((pSrc->height >> nLayer) + 1) << nLayer;

    }
    // create wavelet image
    IplImage * pWavelet = cvCreateImage(size, IPL_DEPTH_32F, pSrc->nChannels);
    size1.width = pSrc->width >> nLayer;
    size1.height = pSrc->height >> nLayer;
    cout << "size = " << size.width << ", " << size.height << endl;
    cout << "size = " << size1.width << ", " << size1.height << endl;
    if (pWavelet) {
        // evaluation of wavelet image
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pSrc, pWavelet, 1, -128); //use linear transformation to convert to array：pWavelet = pSrc*1-128

        // wavelet transformation of color image
        IplImage * pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);

        if (pImage) {
            for (int i = 1; i <= pWavelet->nChannels; i++) {
                cvSetImageCOI(pWavelet, i); //set interesting channel
                cvCopy(pWavelet, pImage, NULL); //pImage is grayscale image，copy every data of channel in pWavelet to pImage
                // 2-D Continuous Wavelet Transform
                DWT(pImage, nLayer, stoi(argv[2])); //DWT every channel
                // 2-D Continuous Wavelet recover
                IDWT(pImage, nLayer);
                cvCopy(pImage, pWavelet, NULL); //save every data after the transformation of channels into the channel of pWavelet
            }
            cvSetImageCOI(pWavelet, 0);
            cvReleaseImage( &pImage);
        }

        // wavelet transformation image
        cvSetImageROI(pWavelet, cvRect(0, 0, pSrc->width, pSrc->height));
        cvConvertScale(pWavelet, pSrc, 1, 128);
        cvResetImageROI(pWavelet); // this line is not necessary but it’s good for the coding habit
        cvReleaseImage( &pWavelet);
    }
    // display image pSrc
    cvNamedWindow("dwt", 1);
    cvShowImage("dwt", pSrc);

    cvSetImageROI(pSrc, cvRect(0, 0, size.width, size.height)); //set source image ROI
    IplImage * pDest = cvCreateImage(size, pSrc->depth, pSrc->nChannels); //set target image
    cvCopy(pSrc, pDest); //copy image
    cvResetImageROI(pDest); //empty ROI after using the source image
    cvSaveImage("lenna_haar_50.png", pDest); //save target image


    cvWaitKey(0);
    cvDestroyWindow("dwt");

    // ...
    cvReleaseImage( &pSrc);

    return 0;
    return 0;
}
#endif
