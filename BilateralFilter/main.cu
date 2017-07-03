
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
	//Ha a help argumentummal indítjuk a programot, ismertetjük a program mûködését.
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
	cudaSetDevice(0);	//TODO:  csekkolni a hibát?????

	float sigma_s, sigma_r;		//a megfelelõ Gauss függvények paraméterei
	int r, threads;				//r: a spatial kernel sugara, threads: a blokkonkénti thread-ek száma adott dimenzióban

	int returnValue = readConfigParameters(argc, argv, sigma_s, sigma_r, r, threads);
	if (returnValue != 0) {
		return returnValue;
	}

	cv::Mat image;						//openCV függvénnyel olvassuk be a képet.
	image = cv::imread(argv[1], 0);		//beolvassuk a képet, 8 bit szürkeárnyalatossá konvertáljuk
	if (!image.data) {
		fprintf(stderr, "Could not open or find the input image\n\n");
		return NO_IMAGE_ERROR;
	}

	int width = image.cols;
	int height = image.rows;
	int imageSize = width * height;
	int n = 2 * r + 1;				//a spatial kernel oldalának hossza
	int spatialKernelSize = n * n;
	int rangeKernelSize = 511;

	float *d_spatialKernel = NULL, *d_rangeKernel = NULL;		//mindent NULL-ra állítunk, mert ha valami hiba van, akkor memóriafoglalás és használat elõtt is
	unsigned char *d_inputImage = NULL, *d_outputImage = NULL;	//felszabadíthatunk egy adott pointert a freeEverything függgvénnyel. 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	//Az összes használt device memóriát lefoglaljuk. Ha hiba van, kilépünk.
	if (!doAllMallocs(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage, spatialKernelSize, rangeKernelSize, imageSize)) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MALLOC_ERROR;
	}
	
	//feltöltjük a spatialKernel és rageKernel tömböket a GPU oldalon.
	//Megj.: ezek kis méretû tömbök, nem biztos, hogy ilyen kevés adat esetén jól járunk a GPU-val.
	//viszont ha CPU-n számítanánk ezeket, akkor még másolni is kéne host --> device, ez is hosszú.
	//Megj.: lehetne stream-eket használni és párhuzamosan futtatni a két kernelt, de nem nyerünk sokat, mert a bilateral filter kernel dominál.
	//ráadásul régebbi GPU-k csak cudaMemcpy-t és kernelhívást tudnak együtt. 
	createSpatialKernel<<<dim3(n, n), 1>>>(d_spatialKernel, r, sigma_s);

	createRangeKernel<<<1, rangeKernelSize>>>(d_rangeKernel, sigma_r);

	//a képe másolása a device-ra.
	if (cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	//a thread-ek száma adott (vagy a felhasználó mondja meg, vagy alapértelmezetten 32).
	//az adott dimenzióhoz tartozó block-ok számát (gridDim.x, gridDim.y)a következõképpen választjuk meg: 
	//pl az X irányú blokk szám (blocksX) legyen a legkisebb olyan szám, melyre threads * blocksX >= width teljesül. 
	//Így biztosítjuk, hogy minden pixelt megprocesszáljunk 
	int blocksX = (width + threads - 1) / threads;
	int blocksY = (height + threads - 1) / threads;

	//a shared memóriában tároljuk a spatialKernel és a rangeKernel tömböt is, a két méretnek az összegét kell megadnunk a kernel híváskor.
	int sharedMemSize = (spatialKernelSize + rangeKernelSize) * sizeof(float);

	bilateralFilter<<<dim3(blocksX, blocksY), dim3(threads, threads), sharedMemSize >>>
		(d_inputImage, d_outputImage, d_spatialKernel, d_rangeKernel, r, width, height);

	if (cudaMemcpy(image.data, d_outputImage, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;		//Mérjük, hogy mennyi ideig tartott a GPU specifikus utasítások végrehajtása.
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Execution time: %3.1f ms\n"
		   "with parameters: sigma_s = %3.1f, sigma_r = %3.1f, spatial kernel radius = %d, number of threads per block dim = %d\n\n",
			elapsedTime, sigma_s, sigma_r, r, threads);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (!cv::imwrite(argv[2], image)) {
		fprintf(stderr, "Failed to save the processed image.\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		return NO_IMAGE_ERROR;
	}

	freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
	return 0;	//csak akkor térünk vissza 0-val, ha minden rendben ment
}