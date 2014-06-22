#include "opencv2\highgui\highgui.hpp"

int PlayMovie(const char* fileName) {
    cvNamedWindow("Movie", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateFileCapture(fileName);
	IplImage* frame;
	while (true) {
		frame = cvQueryFrame(capture);
		if (frame == NULL)
			break;
		cvShowImage("Movie", frame);

		// 'ESC' key for exit
		// Assuming at 30fps, check frame rate to be sure
		char c = cvWaitKey(33);
		if (c == 27)
			break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Movie");
	return 0; 
}