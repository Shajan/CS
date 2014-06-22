#include <iostream>
using namespace std;

extern int TestMain(const char* fileName);
extern int ReadImageMain(const char* fileName);
extern int PlayMovie(const char* fileName);

char* image = "..\\opencv-logo.png";
char* video = "..\\test-video.mov";

int main(int argc, char **argv) {
    const char* fileName = NULL;
    if (argc == 2) {
        fileName = argv[1];
    } else {
        fileName = video;
    }
    
    //return TestMain(fileName);
    //return ReadImageMain(fileName);
    return PlayMovie(fileName);
}
