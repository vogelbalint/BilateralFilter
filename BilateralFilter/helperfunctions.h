#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

#include <stdio.h>  //A FILE* miatt

#define HELP_MESSAGE "The arguments of the BilateralFilter program:\n"\
"BilateralFilter input_image output_image sigma_s sigma_r radius threads\n"\
"where:\n"\
"  -  input_image: full path of the image file you want to process\n"\
"  -  output_image: full path of the file where you want to save the processed image\n"\
"  -  sigma_s: deviation of the spatial Gaussian function. Positive floating point number.\n"\
"  -  sigma_r: deviation of the range Gaussian function. Positive floating point number.\n"\
"  -  radius: radius of the spatial kernel. Integer, greater than zero.\n"\
"             The spatial kernel matrix contains 2*radius+1 columns and rows, the full size is (2*radius+1)^2\n"\
"             Note: you don't have to specify this argument. If you don't specify it, the program computes it so that radius < 2*sigma_s is fulfilled.\n"\
"  -  threads: number of threads per block per dimension (so blockdim.x = blockdim.y = threads). Integer, greater than zero.\n"\
"              Note: you don't have to specify it. Default value is 32.\n\n"


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
