
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math.h"

#define MAX_RANGE_DIFF 255

//Gauss f�ggv�nyt sz�m�t. Fontos: x n�gyzet�t kell �tadni neki, meg a sigma-t.  
__device__ float gauss(float x_square, float sigma)
{
	return expf(- x_square / (2 * sigma * sigma));
}

//A t�rbeli kernelt el�re kisz�m�tom, hogy ne kelljen egy adott pixel eset�n a szumm�z�s egy adott l�p�s�ben exponenci�lis f�ggv�nyt
//sz�m�tani, mert ez dr�ga m�vlet. Egyszer�bb az, ha a lehets�ges �rt�keket kisz�m�tjuk, ezt betessz�k egy m�trixba (illetve egy t�mbbe)
//ezt a t�b�t bet�tj�k a shared mem�ri�ba is innen szedj�k majd el�.
//a spatialKernel t�mb tartalmazza a lehets�ges t�rbeli elt�r�sekhez tartoz� Gauss f�ggv�ny �rt�keket.
//r: a t�rbeli kernel sugara (vagyis k�t pixel k�z�tti legnagyobb t�rbeli elt�r�s, amit figyelembe vesz�nk, r).
//sigma: a sptial Gaiss f�ggv�nyhez tartot� digma.
__global__ void createSpatialKernel(float *spatialKernel, int r, float sigma)
{
	int n = 2 * r + 1;		//a kernel oldal�nak hossza
	int i = blockIdx.x - r;	//oszlop index a spatial kernelben
	int j = blockIdx.y - r;	//sor index a spatial kernelben
	float x_square = (float)(i * i + j * j);
	spatialKernel[blockIdx.x + n * blockIdx.y] = gauss(x_square, sigma);
}


//K�t pixel intenzit�s�nak k�l�nbs�ge 255*2+1 = 511 f�le �rt�k lehet (a legkisebb 0-255 = -255, a legnagyobb 255 - 0 = 255)
//�rdemes ezeket is el�re kisz�m�tani, mert egy adott pixelhez tartoz� G(I_i - I_j) (az inenzt�s k�l�nbs�ghez tartoz� Gauss)
//kisz�m�t�sa k�lts�ges m�velet, 511 pedig nem olyan nagy sz�m. Ez hasonl� az el�z� spatial kernelhez.
//a lehets�ges intenzit�s k�l�nbs�gekhez tartoz� Gauss �rt�keket t�rol� t�mb�t rangeKernel-nek nevezem (nem prec�z).
//az intenzit�s k�l�nbs�g abszol�t �rt�k�ek maximuma MAX_RANGE_DIFF
__global__ void createRangeKernel(float *rangeKernel, float sigma)
{
	//el�sz�r csak a pozit�v delte I -khez tartoz� Gausst sz�m�tjuk ki, mert szimmetrkus a f�ggv�ny
	int tid = threadIdx.x;
	if (tid >= MAX_RANGE_DIFF) {
		int deltaI = threadIdx.x - MAX_RANGE_DIFF;
		rangeKernel[tid] = gauss((float)(deltaI * deltaI), sigma);
	}

	__syncthreads();

	//�tm�soljuk a negat�v �rt�kekhez tartoz� cuccokat
	int last = MAX_RANGE_DIFF * 2;  //=510
	if (tid < MAX_RANGE_DIFF) {
		rangeKernel[tid] = rangeKernel[last - tid];
	}
}


//A bilaterel filtert megval�s�t� cuda kernel.
//es� k�t argumentum: a bemen� is kimen� k�p pixeleinek intenzit�s �rt�keit tartalmaz� t�mb�k
//spatialKernel, rangeKernel: a t�rbeli �s intenzit�sbeli k�l�nbs�gekhez tartoz� Gauss �rtkeket t�rol� t�mb�k.
//Ezekb�l sokszor olvasunk, ez�rt ezeket a shared mem�ri�ba m�soljuk.
//r: a spatial kernel sugara ; width, height: a k�p sz�less�ge �s magass�g, pixelben.
__global__ void bilateralFilter(unsigned char *in, unsigned char *out, float *spatialKernel, float *rangeKernel, int r,
								int width, int height)
{
	int n = 2 * r + 1;			//a spatial kernel oldal�nak hossza
	int spatialKernelSize = n * n;
	extern __shared__ float sharedData[];	//A shared memory t�rolja a spatial kernel �s a rangeKernel �rt�keit is, egym�s ut�n folytonosan 
	float *pSpatialKernel = &sharedData[r * n + r];					//a shared memory spatial kernelt t�rol� r�sz�nek k�zep�re mutat� ponter
	float *pRangeKernel = &sharedData[spatialKernelSize + 255];		//a shared memory range kernelt t�rol� r�sz�nek k�zep�re mutat

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
	int rangeKernelSize = 2 * MAX_RANGE_DIFF + 1;		//=511
	while (index < rangeKernelSize) {
		sharedData[index + spatialKernelSize] = rangeKernel[index];
		index += step;
	}

	__syncthreads();
	//megvagyunk a shared memory felt�lt�s�vel, j�het a l�nyeg:

	int x = threadIdx.x + blockIdx.x * blockDim.x;			//pixel koordin�t�k kisz�m�t�sa
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;			//a pixel intenzit�s�t t�rol� mem�ria indexe az in �s out t�mb�kben

	if (x < width && y < height) {				//csak az �rv�nyes pixeleket n�zz�k
		float summa = 0, weightSumma = 0;
		int intensity = in[offset];				//az adott pixel inenzit�sa

		for (int j = -r; j <= r; ++j) {			//j: sorindex
			int yj = y + j;						//az aktu�lisan vizsg�lt pixel y koordin�t�ja

			for (int i = -r; i <= r; ++i) {		//i: oszlopindex
				int xi = x + i;					//az aktu�lisan vizsg�lt pixel x koordin�t�ja

				if (xi >= 0 && xi < width && yj >= 0 && yj < height) {
					int offsetij = xi + yj * blockDim.x * gridDim.x;	//az xi , yj pixel intenzit�s�t t�rol� mem�ria indexe
					int intensityij = in[offsetij];						//az xi, yj pixel intenzit�sa
					int deltaI = intensityij - intensity;				//az intenzit�sok k�l�nbs�ge
					float temp = pSpatialKernel[i + j * n] * pRangeKernel[deltaI];
					weightSumma += temp;
					summa += temp * intensityij;
				}
			}
		}

		out[offset] = (unsigned char)(summa / weightSumma);		//TODO: ink�bb kerek�tsen, mint lev�gjon
	}
}


