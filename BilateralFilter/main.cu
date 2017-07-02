
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"
#include "helperfunctions.h"


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
		fprintf(stderr, "You don't have a CUDA capable GPU. Buy one! Sorry.\n");
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
	int n = 2 * r + 1;
	int spatialKernelSize = n * n;
	int rangeKernelSize = 511;

	float *d_spatialKernel = NULL, *d_rangeKernel = NULL;
	unsigned char *d_inputImage = NULL, *d_outputImage = NULL;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	if (!doAllMallocs(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage, spatialKernelSize, rangeKernelSize, imageSize)) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MALLOC_ERROR;
	}

	createSpatialKernel<<<dim3(n, n), 1>>>(d_spatialKernel, r, sigma_s);

	createRangeKernel<<<1, rangeKernelSize>>>(d_rangeKernel, sigma_r);

	if (cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	int blocksX = (width + threads - 1) / threads;
	int blocksY = (height + threads - 1) / threads;
	int sharedMemSize = (spatialKernelSize + rangeKernelSize) * sizeof(float);

	bilateralFilter<<<dim3(blocksX, blocksY), dim3(threads, threads), sharedMemSize >>>
		(d_inputImage, d_outputImage, d_spatialKernel, d_rangeKernel, r, width, height);

	if (cudaMemcpy(image.data, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Execution time: %f ms\n"
			"with parameters: sigma_s = %f, sigma_r = %f, spatial kernel radius = %d, number of threads per block dim = %d\n\n",
			elapsedTime, sigma_s, sigma_r, r, threads);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (!cv::imwrite(argv[2], image)) {
		fprintf(stderr, "Fail saving the processed image.\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return NO_IMAGE_ERROR;
	}

	freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
	return 0;	//csak akkor tér vissza 0-val, ha minden rendben ment
}