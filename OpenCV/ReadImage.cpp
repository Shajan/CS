#include "opencv2\highgui\highgui.hpp"

int ReadImageMain(const char* fileName) {
    IplImage* img = cvLoadImage(fileName);
    cvNamedWindow("ReadImage", CV_WINDOW_AUTOSIZE);
    cvShowImage("ReadImage", img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvDestroyWindow("ReadImage");
    return 0;
}