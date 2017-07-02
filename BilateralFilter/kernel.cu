

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math.h"


__device__ float gauss(float x_square, float sigma)
{
	return expf(-x_square / (2 * sigma * sigma));
}

//ek�tte le kell foglalni cudaMalloc-kal, spatialKernel egy device pointer
__global__ void createSpatialKernel(float *spatialKernel, int r, float sigma)
{
	int n = 2 * r + 1;		//a kernel oldal�nak hossza
	int i = threadIdx.x - r;	//oszlop index a spatial kernelben
	int j = threadIdx.y - r;	//sor index a spatial kernelben
	float x_square = (float)(i * i + j * j);
	spatialKernel[threadIdx.x + n * threadIdx.y] = gauss(x_square, sigma);
}

//rangeKernel: m�rete: 255 * 2 + 1 = 511
//TODO magic numbers, g�nyol�s elt�ntet�se
__global__ void createRangeKernel(float *rangeKernel, float sigma)
{
	int tid = threadIdx.x;
	if (tid >= 255) {
		int deltaI = threadIdx.x - 255;
		rangeKernel[tid] = gauss((float)(deltaI * deltaI), sigma);
	}

	__syncthreads();

	if (tid < 255) {
		rangeKernel[tid] = rangeKernel[510 - tid];
	}
}

__global__ void bilateralFilter(unsigned char *in, unsigned char *out, float *spatialKernel, float *rangeKernel, int r,
	int width, int height)
{
	int n = 2 * r + 1;			//a spatial kernel oldal�nak hossza
	int spatialKernelSize = n * n;
	extern __shared__ float sharedData[];
	float *pSpatialKernel = &sharedData[r * n + r];					//a shared memory spatial kernelt t�rol� r�sz�nek k�zep�re mutat
	float *pRangeKernel = &sharedData[spatialKernelSize + 255];	//a shared memory range kernelt t�rol� r�sz�nek k�zep�re mutat

	//A shared memory felt�lt�se:
	//1. minden thread �tm�solja a megfelel� spatialKernel elemet
	int index = threadIdx.x + blockDim.x * threadIdx.y;
	int step = blockDim.x * blockDim.y;		//az �sszes thread sz�ma a blockban
	while (index < spatialKernelSize) {
		sharedData[index] = spatialKernel[index];
		index += step;
	}

	//2. minden thread �tm�solja a megfelel� rangeKernel elemet
	index = threadIdx.x + blockDim.x * threadIdx.y;
	while (index < 511) {
		sharedData[index + spatialKernelSize] = rangeKernel[index];
		index += step;
	}

	__syncthreads();
	//megvagyunk a shared memory felt�lt�s�vel, j�het a l�nyeg:

	int x = threadIdx.x + blockIdx.x * blockDim.x;			//pixel koordin�t�k kisz�m�t�sa
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;			//a pixel intenzit�s�t t�rol� mem�ria indexe

	if (x < width && y < height) {
		float summa = 0, weightSumma = 0;
		int intensity = in[offset];				//az adott pixel inenzit�sa

		for (int j = -r; j <= r; ++j) {			//j: sorindex
			int yj = y + j;						//az aktu�lisan vizsg�lt pixel y koordin�t�ja

			for (int i = -r; i <= r; ++i) {		//i: oszlopindex
				int xi = x + i;					//az aktu�lisan vizsg�lt pixel x koordin�t�ja
												//printf("%d %d %d %d\n", x, y, xi, yj);

				if (xi >= 0 && xi < width && yj >= 0 && yj < height) {
					int offsetij = xi + yj * blockDim.x * gridDim.x;	//az xi , yj pixel intenzit�s�t t�rol� mem�ria indexe

																		//int offsetij = xi + yj * blockDim.x * gridDim.x;
																		//if (offsetij < height *width) {

					int intensityij = in[offsetij];						//az xi, yj pixel intenzit�sa
					int deltaI = intensityij - intensity;
					float temp = pSpatialKernel[i + j * n] * pRangeKernel[deltaI];
					//float temp = sharedData[(i + r) + (j + r) * n] * sharedData[spatialKernelSize + 255 + deltaI];
					weightSumma += temp;
					summa += temp * intensityij;

					//printf("%f %f\n", summa, weightSumma);
				}
			}
		}
		//printf("%f %f\n", summa, weightSumma);
		//out[offset] = (in[offset] + 100 < 256) ? (in[offset] + 100) : 255;
		out[offset] = (unsigned char)(summa / weightSumma);
		//out[offset] = (in[offset] + 100 < 256) ? (in[offset] + 100) : 255;
		//out[offset] = (summa > 255) ? 255 : ((summa < 0) ? 128 : summa);
	}
}


