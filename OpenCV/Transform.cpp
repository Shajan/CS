#include "opencv2\opencv.hpp"

int GaussianFilter(const char* fileName) {
    IplImage* input = cvLoadImage(fileName);
    IplImage* output = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 3);

    cvSmooth(input, output, CV_GAUSSIAN, 23, 23);
    IplImage* img = input;

    while (true) {
        cvNamedWindow("Display", CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
        cvShowImage("Display", img);
        // Break on 'ESC'
        char c = cvWaitKey();
        if (c == 27)
            break;
        // Toggle
        img = (img == input) ? output : input;
    }

    cvReleaseImage(&output);    
    cvReleaseImage(&input);
    cvDestroyWindow("Output");
    cvDestroyWindow("Input");

    return 0;    
}