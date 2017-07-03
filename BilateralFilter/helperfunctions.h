#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <stdio.h>  //A FILE* miatt

#define NO_DEVICE_ERROR 1
#define ARGUMENT_ERROR 2
#define NO_IMAGE_ERROR 3
#define CUDA_MALLOC_ERROR 4
#define CUDA_MEMCPY_ERROR 5

void printHelpMessage(FILE *stream);

int readConfigParameters(int argc, char **argv, float & sigma_s, float & sigma_r, int & r, int & threads);

bool doAllMallocs(float * & d_spatialKernel, float * & d_rangeKernel, unsigned char * & d_inputImage, unsigned char * & d_outputImage,
				int spatialKernelSize, int rangeKernelSize, int imageSize);

void freeEverything(float *d_spatialKernel, float *d_rangeKernel, unsigned char *d_inputImage, unsigned char *d_outputImage);

#endif  //HELPERFUNCTIONS_H
