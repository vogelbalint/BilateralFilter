


#ifndef KERNEL_H
#define KERNEL_H

__global__ void createSpatialKernel(float *spatialKernel, int r, float sigma);

__global__ void createRangeKernel(float *rangeKernel, float sigma);

__global__ void bilateralFilter(unsigned char *in, unsigned char *out, float *spatialKernel, float *rangeKernel, int r,
								int width, int height);


#endif //KERNEL_H


