
// CLE 24'25
#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
// __global__ void helloKernel(){
// 	printf("Hello from thread(%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
// }

__global__ void convolution(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn);

__global__ void non_maximum_supression(const pixel_t *after_Gx, const pixel_t * after_Gy, const pixel_t *G, pixel_t *nnnnms, const int nx, const int ny);

__global__ void first_edges(const pixel_t *nms, pixel_t *reference,	const int nx, const int ny, const int tmax);


__global__ void convolution(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn){

	// index of the pixel that the thread will process
	int m = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.y + blockDim.y * blockIdx.y;

	const int khalf = kn / 2;

	// does not process pixels that are borders
	if (m >= khalf && m < nx - khalf && n >= khalf && n < ny - khalf){
		float pixel = 0.0;
		size_t c = 0;
		for (int j = -khalf; j <= khalf; j++)
			for (int i = -khalf; i <= khalf; i++){
				pixel += in[(n -j) * nx + m - i] * kernel[c];
				c++;
			}
		out[n * nx + m] = (pixel_t)pixel;
	}
}

__global__ void non_maximum_supression(const pixel_t *after_Gx, const pixel_t * after_Gy, const pixel_t *G, pixel_t *nnnnms, const int nx, const int ny){
	int m = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.y + blockDim.y * blockIdx.y;

	if(m > 0 &&  m < nx - 1 && n > 0 && n < ny -1) { 
		const int c = m + nx * n;
		const int nn = c - nx;
		const int ss = c + nx;
		const int ww = c + 1;
		const int ee = c - 1;
		const int nw = nn + 1;
		const int ne = nn - 1;
		const int sw = ss + 1;
		const int se = ss - 1;
	
		const float dir = (float)(fmod(atan2(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI) * 8;
	
		if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
				 G[c] > G[ww]) || // 0 deg
				((dir > 1 && dir <= 3) && G[c] > G[nw] &&
				 G[c] > G[se]) || // 45 deg
				((dir > 3 && dir <= 5) && G[c] > G[nn] &&
				 G[c] > G[ss]) || // 90 deg
				((dir > 5 && dir <= 7) && G[c] > G[ne] &&
				 G[c] > G[sw]))   // 135 deg
			nms[c] = G[c];
		else
			nms[c] = 0;
	}

}

__global__ void first_edges(const pixel_t *nms, pixel_t *reference,	const int nx, const int ny, const int tmax){
	int m = threadIdx.x + blockDim.x * blockIdx.x;
	int n = threadIdx.y + blockDim.y * blockIdx.y;

	size_t c = m;
	if(m > 0 &&  m < nx - 1 && n > 0 && n < ny - 1) {
		if(nms[c] >= tmax){
			reference[c] = MAX_BRIGHTNESS;
		}
		
	}
	
}

// canny edge detector code to run on the GPU
void cannyDevice( const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int * h_odata)
{

	//from the class cuda exercises
  // set up device
  int dev = 0;

  cudaDeviceProp deviceProp;
  CHECK (cudaGetDeviceProperties (&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK (cudaSetDevice (dev)); // a gpu que vou utilizar


	//test cuda devide is running

	blockDimX = 16;
	blockDimY = 16;
	//Every pixel is proccess by at least one thread
	gridDimX = (nx + blockDimX - 1) / blockDimX;
	gridDimY = (ny + blockDimY - 1) / blockDimY;n
	dim3 block (16, 16);
	dim3 grid (gridDimX, gridDimY);
	// helloKernel<<<grid, block>>>();
	// cudaDeviceSynchronize();

	//LAST EXERCISE
	//gaussianFilter - probably create a function for this
	/*const int n = 2 * (int)(2 * sigma) + 3;
	const float mean = (float)floor(n / 2.0);
	float kernel[n * n];

	fprintf(stderr, "gaussian_filter: kernel size %d, sigma=%g\n", n, sigma);

	size_t c = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			kernel[c] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) /sigma, 2.0))) / (2 * M_PI * sigma * sigma);
			c++
		}
	
	assert(n % 2 == 1);
	assert(nx > n && ny > n);
	convolution<<<grid, block>>>(in, out, kernel, nx, ny, n);

	pixel_t max, min;
	min_max<<<grid, block>>>(out, nx, ny, &min, &max);
	*/
	
}
