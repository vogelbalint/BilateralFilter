

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{


Mat image;
image = imread("matterhorn.jpg", 0); // Read the file

if (!image.data) // Check for invalid input
{
cout << "Could not open or find the image" << std::endl;
return -1;
}

imwrite("matterhorn2.jpg", image);

return 0;
}
