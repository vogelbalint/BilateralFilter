#include "helperfunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#define THREADS_DEFAULT 32

void printHelpMessage(FILE *stream)
{
	fprintf(stream, HELP_MESSAGE);
}

int readConfigParameters(int argc, char **argv, float & sigma_s, float & sigma_r, int & r, int & threads)
{
	//Parancssori paraméterek feldolgozása
	//Ellenõrizzük a paraméterek számát:
	if (!(argc >= 5 && argc <= 7)) {
		fprintf(stderr, "Number of arguments is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}

	//beolvassuk a sigma_s, sigma_r, r threads paramétereket
	//sigma_s :
	double temp_d = atof(argv[3]);
	if (temp_d == 0.0) {
		fprintf(stderr, "Argument sigma_s is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}
	else {
		sigma_s = temp_d;
	}

	//sigma_r :
	temp_d = atof(argv[4]);
	if (temp_d == 0.0) {
		fprintf(stderr, "Argument sigma_r is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}
	else {
		sigma_r = temp_d;
	}

	//r beolvasása. Ennek megadása opcionális
	if (argc >= 6) {
		r = atoi(argv[5]);
		if (r <= 0) {
			fprintf(stderr, "The radius of the spatial kernel is incorrect.\n\n");
			printHelpMessage(stderr);
			return ARGUMENT_ERROR;
		}
	}
	else {  //ha nincs megadva r, akkor kiszámítjuk sigma_s alapján. r <= 2 * sigma_s teljesül
		r = (int)(2 * sigma_s);
	}

	//threads beovasása
	if (argc == 7) {
		threads = atoi(argv[6]);
		if (threads <= 0) {
			fprintf(stderr, "The number of threads per block dimension is incorrect.\n\n");
			printHelpMessage(stderr);
			return ARGUMENT_ERROR;
		}
	}
	else {
		threads = THREADS_DEFAULT;
	}

	return 0;
}

void freeEverything(float *d_spatialKernel, float *d_rangeKernel, unsigned char *d_inputImage, unsigned char *d_outputImage)
{
	cudaFree(d_spatialKernel);
	cudaFree(d_rangeKernel);
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
}

bool doAllMallocs(float * & d_spatialKernel, float * & d_rangeKernel, unsigned char * & d_inputImage, unsigned char * & d_outputImage,
					int spatialKernelSize, int rangeKernelSize, int imageSize)
{
	if (cudaMalloc((void**)&d_spatialKernel, spatialKernelSize * sizeof(float)) != cudaSuccess) {
		d_spatialKernel = NULL;
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return false;
	}

	if (cudaMalloc((void**)&d_rangeKernel, rangeKernelSize * sizeof(float)) != cudaSuccess) {
		d_rangeKernel = NULL;
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return false;
	}
	
	if (cudaMalloc((void**)&d_inputImage, imageSize * sizeof(unsigned char)) != cudaSuccess) {
		d_inputImage = NULL;
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return false;
	}

	if (cudaMalloc((void**)&d_outputImage, imageSize * sizeof(unsigned char)) != cudaSuccess) {
		d_outputImage = NULL;
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return false;
	}

	return true;
}
