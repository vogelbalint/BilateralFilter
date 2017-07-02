
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "helperfunctions.h"

//using namespace cv;
//using namespace std;

int main(int argc, char** argv)
{
	if (argc == 2 && strcmp("help", argv[1]) == 0) {
		printHelpMessage(stdout);
		return 0;
	}
	//Megnézzük, hogy van-e megfelelõ GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "You don't have a CUDA enabled GPU. Buy one! Sorry.\n");
		return NO_DEVICE_ERROR;
	}
	cudaSetDevice(0);	//TODO:  csekkolni a hibát!!!

	float sigma_s, sigma_r;
	int r, threads;

	int returnValue = readConfigParameters(argc, argv, sigma_s, sigma_r, r, threads);
	if (returnValue != 0) {
		return returnValue;
	}

	cv::Mat image;
	image = cv::imread(argv[1], 0);		//beolvassuk a képet, 8 bit szürkeárnyalatossá konvertáljuk
	if (!image.data) {
		fprintf(stderr, "Could not open or find the input image\n\n");
		return NO_IMAGE_ERROR;
	}

	int width = image.cols;
	int height = image.rows;
	int imageSize = width * height;
	int spatialKernelSize = (2 * r + 1)*(2 * r + 1);
	int rangeKernelSize = 511;

	float *d_spatialKernel = NULL, *d_rangeKernel = NULL;
	unsigned char *d_inputImage = NULL, *d_outputImage = NULL;

	if (!doAllMallocs(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage, spatialKernelSize, rangeKernelSize, imageSize)) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		return CUDA_MALLOC_ERROR;
	}

	createSpatialKernel<<<1, spatialKernelSize>>>(d_spatialKernel, r, sigma_s);
	createRangeKernel<<<1, 511>>>(d_rangeKernel, sigma_r);

	cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int blocksX = (width + threads - 1) / threads;
	int blocksY = (height + threads - 1) / threads;

	int sharedMemSize = (spatialKernelSize + rangeKernelSize) * sizeof(float);

	bilateralFilter<<<dim3(blocksX, blocksY), dim3(threads, threads), sharedMemSize >>>
		(d_inputImage, d_outputImage, d_spatialKernel, d_rangeKernel, r, width, height);

	cudaMemcpy(image.data, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cv::imwrite(argv[2], image);

	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
	cudaFree(d_spatialKernel);
	cudaFree(d_rangeKernel);
	return 0;
}