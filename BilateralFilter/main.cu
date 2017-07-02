
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	cudaSetDevice(0);

	Mat image;
	image = imread("matterhorn.jpg", 0); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	int width = image.cols;
	int height = image.rows;
	int imageSize = width * height;

	int r = 7;
	int spatialKernelSize = (2 * r + 1)*(2 * r + 1);
	float sigma_s = 20.0f, sigma_r = 30.0f;

	float *d_spatialKernel, *d_rangeKernel;
	cudaMalloc((void**)&d_spatialKernel, spatialKernelSize * sizeof(float));
	cudaMalloc((void**)&d_rangeKernel, 511 * sizeof(float));
	createSpatialKernel<<<1, spatialKernelSize>>>(d_spatialKernel, r, sigma_s);
	createRangeKernel<<<1, 511>>>(d_rangeKernel, sigma_r);

	unsigned char *d_inputImage;
	cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char));
	cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	unsigned char *d_outputImage;
	cudaMalloc((void**)&d_outputImage, imageSize * sizeof(unsigned char));

	int threads = 32;
	int blocksx = (width + threads - 1) / threads;
	int blocksy = (height + threads - 1) / threads;

	int sharedMemSize = (spatialKernelSize + 511) * sizeof(float);

	bilateralFilter<<<dim3(blocksx, blocksy), dim3(threads, threads), sharedMemSize >>>
		(d_inputImage, d_outputImage, d_spatialKernel, d_rangeKernel, r, width, height);

	cudaMemcpy(image.data, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	imwrite("matterhorn2.jpg", image);

	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
	cudaFree(d_spatialKernel);
	cudaFree(d_rangeKernel);
	return 0;
}