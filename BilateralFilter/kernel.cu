
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "math.h"

#define MAX_RANGE_DIFF 255

//Gauss függvényt számít. Fontos: x négyzetét kell átadni neki, meg a sigma-t.  
__device__ float gauss(float x_square, float sigma)
{
	return expf(- x_square / (2 * sigma * sigma));
}

//A térbeli kernelt elõre kiszámítom, hogy ne kelljen egy adott pixel esetén a szummázás egy adott lépésében exponenciális függvényt
//számítani, mert ez drága mûvlet. Egyszerûbb az, ha a lehetséges értékeket kiszámítjuk, ezt betesszük egy mátrixba (illetve egy tömbbe)
//ezt a tömböt betötjük a shared memóriába is innen szedjük majd elõ az értékeket.
//a spatialKernel tömb tartalmazza a lehetséges térbeli eltérésekhez tartozó Gauss függvény értékeket.
//r: a térbeli kernel sugara (vagyis két pixel közötti legnagyobb térbeli eltérés, amit figyelembe veszünk, r).
//sigma: a spatial Gaussian-hoz tartotó sigma.
__global__ void createSpatialKernel(float *spatialKernel, int r, float sigma)
{
	int n = 2 * r + 1;		//a kernel oldalának hossza
	int i = blockIdx.x - r;	//oszlop index a spatial kernelben
	int j = blockIdx.y - r;	//sor index a spatial kernelben
	float x_square = (float)(i * i + j * j);
	spatialKernel[blockIdx.x + n * blockIdx.y] = gauss(x_square, sigma);
}


//Két pixel intenzitásának különbsége 255*2+1 = 511 féle érték lehet (a legkisebb 0-255 = -255, a legnagyobb 255 - 0 = 255)
//érdemes az ezekhez tartozó Gauss értékeket is kiszámítani, mert adott két pixelhez tartozó G(I_i - I_j) (az inenztás különbséghez tartozó Gauss)
//kiszámítása költséges mûvelet, 511 pedig nem olyan nagy szám. Ez hasonló az elõzõ spatial kernelhez.
//a lehetséges intenzitás különbségekhez tartozó Gauss értékeket tároló tömböt rangeKernel-nek nevezem (nem precíz).
//az intenzitás különbség abszolút értékének maximuma MAX_RANGE_DIFF
__global__ void createRangeKernel(float *rangeKernel, float sigma)
{
	//elõször csak a pozitív delte I -khez tartozó Gausst számítjuk ki, mert szimmetrikus a függvény
	int tid = threadIdx.x;
	if (tid >= MAX_RANGE_DIFF) {
		int deltaI = tid - MAX_RANGE_DIFF;
		rangeKernel[tid] = gauss((float)(deltaI * deltaI), sigma);
	}

	__syncthreads();

	//átmásoljuk a negatív intenzitás különbség értkekhez tartozó Gauss értékeket
	int last = MAX_RANGE_DIFF * 2;  //=510
	if (tid < MAX_RANGE_DIFF) {
		rangeKernel[tid] = rangeKernel[last - tid];
	}
}


//A bilaterel filtert megvalósító cuda kernel.
//esõ két argumentum: a bemenõ is kimenõ kép pixeleinek intenzitás értékeit tartalmazó tömbök
//spatialKernel, rangeKernel: lehetséges a térbeli és intenzitásbeli különbségekhez tartozó Gauss értékeket tároló tömbök.
//Ezekbõl sokszor olvasunk, ezért ezeket a shared memóriába másoljuk.
//r: a spatial kernel sugara ; width, height: a kép szélessége és magasság, pixelben.
__global__ void bilateralFilter(unsigned char *in, unsigned char *out, float *spatialKernel, float *rangeKernel, int r,
								int width, int height)
{
	int n = 2 * r + 1;			//a spatial kernel oldalának hossza
	int spatialKernelSize = n * n;
	extern __shared__ float sharedData[];	//A shared memory tárolja a spatial kernel és a rangeKernel értékeit is, egymás után folytonosan 
	float *pSpatialKernel = &sharedData[r * n + r];					//a shared memory spatial kernelt tároló részének közepére mutató pointer
	float *pRangeKernel = &sharedData[spatialKernelSize + MAX_RANGE_DIFF];		//a shared memory range kernelt tároló részének közepére mutat

	//A shared memory feltöltése:
	//1. minden thread átmásolja a megfelelõ spatialKernel elemet
	int index = threadIdx.x + blockDim.x * threadIdx.y;
	int step = blockDim.x * blockDim.y;		//az összes thread száma a blockban
	while (index < spatialKernelSize) {
		sharedData[index] = spatialKernel[index];
		index += step;
	}

	//2. minden thread átmásolja a megfelelõ rangeKernel elemet
	index = threadIdx.x + blockDim.x * threadIdx.y;
	int rangeKernelSize = 2 * MAX_RANGE_DIFF + 1;		//=511
	while (index < rangeKernelSize) {
		sharedData[index + spatialKernelSize] = rangeKernel[index];
		index += step;
	}

	__syncthreads();
	//megvagyunk a shared memory feltöltésével, jöhet a lényeg:

	int x = threadIdx.x + blockIdx.x * blockDim.x;			//pixel koordináták kiszámítása
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < width && y < height) {				//csak az érvényes pixeleket nézzük
		int offset = x + y * width;				//a pixel intenzitását tároló memória indexe az in és out tömbökben
		float summa = 0.0f, weightSumma = 0.0f;
		int intensity = in[offset];				//az adott pixel intenzitása

		for (int j = -r; j <= r; ++j) {			//j: sorindex
			int yj = y + j;						//az aktuálisan vizsgált pixel y koordinátája

			for (int i = -r; i <= r; ++i) {		//i: oszlopindex
				int xi = x + i;					//az aktuálisan vizsgált pixel x koordinátája

				if (xi >= 0 && xi < width && yj >= 0 && yj < height) {
					int offsetij = xi + yj * width;						//az xi , yj pixel intenzitását tároló memória indexe
					int intensityij = in[offsetij];						//az xi, yj pixel intenzitása
					int deltaI = intensityij - intensity;				//az intenzitások különbsége
					float temp = pSpatialKernel[i + j * n] * pRangeKernel[deltaI];
					weightSumma += temp;
					summa += temp * intensityij;
				}
			}
		}
		
		out[offset] = (weightSumma == 0.0f) ? 0 : ((unsigned char)(summa / weightSumma));		//TODO: inkább kerekítsen, mint levágjon
	}
}


