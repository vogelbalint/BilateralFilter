#include "helperfunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

#define THREADS_DEFAULT 32

//makr�, a sztring a program m�k�d�s�t ismerteti. 
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

//Ismerteti a program m�k�d�s�t. stdout-ra vagy stderr-re �rja az �zenetet.
//Ha a program a help argumentummal lett megh�vva, az sdtin-re �r
//ha a parancssori arhumentumok rosszul letek megadva, akkor is ki�r�dik az �zenet, de ekkor az stderr-re.    
void printHelpMessage(FILE *stream)
{
	fprintf(stream, HELP_MESSAGE);
}

//Beolvassa �s feldolgozza a parancssori param�tereket.
//Ha valami rosszul lett megadva, visszaad egy hibak�dot.
//A mainb�l h�vjuk a f�ggv�nyt, a main-ben deklar�lt v�ltoz�kat �ll�tjuk, ez�rt ezeket referenciak�nt vessz�k �t. 
int readConfigParameters(int argc, char **argv, float & sigma_s, float & sigma_r, int & r, int & threads)
{
	//Parancssori param�terek feldolgoz�sa
	////ha valam nem j�, akkor mindig megmondjuk, mi a hiba, ki�rjuk a program m�k�d�s�t bemutat� sztinget �s visszat�r�nk egy hibak�ddal. 
	//Ellen�rizz�k a param�terek sz�m�t. Az utold� k�t param�ter, r �s threads opcion�lisak.
	if (!(argc >= 5 && argc <= 7)) {
		fprintf(stderr, "Number of arguments is incorrect.\n\n");
		printHelpMessage(stderr);
		return ARGUMENT_ERROR;
	}

	//beolvassuk a sigma_s, sigma_r, r threads param�tereket
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

	//r beolvas�sa. Ennek megad�sa opcion�lis
	if (argc >= 6) {
		r = atoi(argv[5]);
		if (r <= 0) {
			fprintf(stderr, "The radius of the spatial kernel is incorrect.\n\n");
			printHelpMessage(stderr);
			return ARGUMENT_ERROR;
		}
	}
	else {  //ha nincs megadva r, akkor kisz�m�tjuk sigma_s alapj�n. r <= 2 * sigma_s teljes�l
		r = (int)(2 * sigma_s);
	}

	//threads beovas�sa
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

//A mainb�l h�vjuk, felszabad�tja az �sszes device pointer-t. Mindig h�vjuk, ha valami hiba van �s ki kell takar�tani.
//Figyelni kell, hogy a main-ben a pointereket deklar�l�skor NULL-ra �ll�tsuk. A device null pointer-re lehet h�vni cudaFree-t. 
void freeEverything(float *d_spatialKernel, float *d_rangeKernel, unsigned char *d_inputImage, unsigned char *d_outputImage)
{
	cudaFree(d_spatialKernel);
	cudaFree(d_rangeKernel);
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);
}

//A main-b�l h�vjuk, lefoglalja a mem�ri�t az �sszes device pointernek, amit haszn�lunk. Ha gond van, mindent felszabad�t �s false-szal t�r vissza.
//Ha minden ok, akkor true-val t�r vissza.
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
