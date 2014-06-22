#include <iostream>
using namespace std;

int render(const char* fileName);
int transform(const char* fileName);

char* image = "..\\test-image.jpg";
char* video = "..\\test-video.mov";

int main(int argc, char **argv) {
    const char* fileName = NULL;
    if (argc == 2) {
        fileName = argv[1];
    } else {
        //fileName = video;
        fileName = image;
    }
    //return render(fileName);
    return transform(fileName);
}

extern int TestMain(const char* fileName);
extern int ReadImageMain(const char* fileName);
extern int PlayMovie(const char* fileName);
extern int PlayMovieWithSlider(const char* fileName);

int render(const char* fileName) {
    //return TestMain(fileName);
    //return ReadImageMain(fileName);
    //return PlayMovie(fileName);
    return PlayMovieWithSlider(fileName);
}

extern int GaussianFilter(const char* fileName);

int transform(const char* fileName) {
    return GaussianFilter(fileName);
}
