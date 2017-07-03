
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
	//Ha a help argumentummal ind�tjuk a programot, ismertetj�k a program m�k�d�s�t.
	if (argc == 2 && strcmp("help", argv[1]) == 0) {
		printHelpMessage(stdout);
		return 0;
	}
	//Megn�zz�k, hogy van-e megfelel� GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "You don't have a CUDA capable GPU. Buy one! Sorry.\n");
		return NO_DEVICE_ERROR;
	}
	cudaSetDevice(0);	//TODO:  csekkolni a hib�t?????

	float sigma_s, sigma_r;		//a megfelel� Gauss f�ggv�nyek param�terei
	int r, threads;				//r: a spatial kernel sugara, threads: a blokkonk�nti thread-ek sz�ma adott dimenzi�ban

	int returnValue = readConfigParameters(argc, argv, sigma_s, sigma_r, r, threads);
	if (returnValue != 0) {
		return returnValue;
	}

	cv::Mat image;						//openCV f�ggv�nnyel olvassuk be a k�pet.
	image = cv::imread(argv[1], 0);		//beolvassuk a k�pet, 8 bit sz�rke�rnyalatoss� konvert�ljuk
	if (!image.data) {
		fprintf(stderr, "Could not open or find the input image\n\n");
		return NO_IMAGE_ERROR;
	}

	int width = image.cols;
	int height = image.rows;
	int imageSize = width * height;
	int n = 2 * r + 1;				//a spatial kernel oldal�nak hossza
	int spatialKernelSize = n * n;
	int rangeKernelSize = 511;

	float *d_spatialKernel = NULL, *d_rangeKernel = NULL;		//mindent NULL-ra �ll�tunk, mert ha valami hiba van, akkor mem�riafoglal�s �s haszn�lat el�tt is
	unsigned char *d_inputImage = NULL, *d_outputImage = NULL;	//felszabad�thatunk egy adott pointert a freeEverything f�gggv�nnyel. 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	//Az �sszes haszn�lt device mem�ri�t lefoglaljuk. Ha hiba van, kil�p�nk.
	if (!doAllMallocs(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage, spatialKernelSize, rangeKernelSize, imageSize)) {
		fprintf(stderr, "cudaMalloc failed!\n\n");
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MALLOC_ERROR;
	}
	
	//felt�ltj�k a spatialKernel �s rageKernel t�mb�ket a GPU oldalon.
	//Megj.: ezek kis m�ret� t�mb�k, nem biztos, hogy ilyen kev�s adat eset�n j�l j�runk a GPU-val.
	//viszont ha CPU-n sz�m�tan�nk ezeket, akkor m�g m�solni is k�ne host --> device, ez is hossz�.
	//Megj.: lehetne stream-eket haszn�lni �s p�rhuzamosan futtatni a k�t kernelt, de nem nyer�nk sokat, mert a bilateral filter kernel domin�l.
	//r�ad�sul r�gebbi GPU-k csak cudaMemcpy-t �s kernelh�v�st tudnak egy�tt. 
	createSpatialKernel<<<dim3(n, n), 1>>>(d_spatialKernel, r, sigma_s);

	createRangeKernel<<<1, rangeKernelSize>>>(d_rangeKernel, sigma_r);

	//a k�pe m�sol�sa a device-ra.
	if (cudaMemcpy(d_inputImage, image.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n\n");
		freeEverything(d_spatialKernel, d_rangeKernel, d_inputImage, d_outputImage);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		return CUDA_MEMCPY_ERROR;
	}

	//a thread-ek sz�ma adott (vagy a felhaszn�l� mondja meg, vagy alap�rtelmezetten 32).
	//az adott dimenzi�hoz tartoz� block-ok sz�m�t (gridDim.x, gridDim.y)a k�vetkez�k�ppen v�lasztjuk meg: 
	//pl az X ir�ny� blokk sz�m (blocksX) legyen a legkisebb olyan sz�m, melyre threads * blocksX >= width teljes�l. 
	//�gy biztos�tjuk, hogy minden pixelt megprocessz�ljunk 
	int blocksX = (width + threads - 1) / threads;
	int blocksY = (height + threads - 1) / threads;

	//a shared mem�ri�ban t�roljuk a spatialKernel �s a rangeKernel t�mb�t is, a k�t m�retnek az �sszeg�t kell megadnunk a kernel h�v�skor.
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

	float elapsedTime;		//M�rj�k, hogy mennyi ideig tartott a GPU specifikus utas�t�sok v�grehajt�sa.
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
	return 0;	//csak akkor t�r�nk vissza 0-val, ha minden rendben ment
}