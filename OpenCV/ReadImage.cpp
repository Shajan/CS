#include "opencv2\highgui\highgui.hpp"

int ReadImageMain(int argc, char** argv) {
  IplImage* img = cvLoadImage(argv[1]);
  cvNamedWindow("ReadImage", CV_WINDOW_AUTOSIZE);
  cvShowImage("ReadImage", img);
  cvWaitKey(0);
  cvReleaseImage(&img);
  cvDestroyWindow("ReadImage");
  return 0;
}