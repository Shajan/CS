#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int testMain(int argc, char** argv)
{
    if (argc != 2) {
        cout <<" Usage: Test <FileName>" << endl;
        return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR);

    // Check for invalid input
    if (!image.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Create a window for display.
    namedWindow("Test Window", WINDOW_AUTOSIZE);

    // Show image inside the window
    imshow("Test Window", image);

    // Wait for a keystroke in the window
    waitKey(0);

    return 0;
}
