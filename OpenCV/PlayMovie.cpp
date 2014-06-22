#include "opencv2\opencv.hpp"

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

int g_slider_position = 0;
CvCapture* g_capture = NULL;

void onTrackbarSlide(int pos) {
    cvSetCaptureProperty(g_capture, CV_CAP_PROP_POS_FRAMES, pos);
}

int PlayMovieWithSlider(const char* fileName) {
    cvNamedWindow("MovieWithSlider", CV_WINDOW_AUTOSIZE);
    g_capture = cvCreateFileCapture(fileName);
    int frames = (int) cvGetCaptureProperty(g_capture, CV_CAP_PROP_FRAME_COUNT);
    if (frames != 0) {
        cvCreateTrackbar("Position", "MovieWithSlider", &g_slider_position, frames, onTrackbarSlide);
    }
    IplImage* frame;
    while (true) {
        frame = cvQueryFrame(g_capture);
        if (frame == NULL)
            break;
        cvShowImage("MovieWithSlider", frame);
        
        // Move the slider
        int pos = (int) cvGetCaptureProperty(g_capture, CV_CAP_PROP_POS_FRAMES);
        cvSetTrackbarPos("Position", "MovieWithSlider", pos);

        // 'ESC' key for exit
        // Assuming at 30fps, check frame rate to be sure
        char c = cvWaitKey(33);
        if (c == 27)
            break;
    }
    cvReleaseCapture(&g_capture);
    g_capture = NULL;
    cvDestroyWindow("MovieWithSlider");
    return 0;
}
