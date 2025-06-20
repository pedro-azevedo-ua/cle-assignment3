// CLE 24'25
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
// __global__ void helloKernel(){
// 	printf("Hello from thread(%d, %d) in block (%d, %d)\n", threadIdx.x,
// threadIdx.y, blockIdx.x, blockIdx.y);
// }
//
typedef int pixel_t;

__global__ void convolution_cuda(const pixel_t *in, pixel_t *out,
                                 const float *kernel, const int nx,
                                 const int ny, const int kn);

__global__ void non_maximum_supression_cuda(const pixel_t *after_Gx,
                                            const pixel_t *after_Gy,
                                            const pixel_t *G, pixel_t *nms,
                                            const int nx, const int ny);

__global__ void first_edges_cuda(const pixel_t *nms, pixel_t *reference,
                                 const int nx, const int ny, const int tmax);

__global__ void hysteresis_edges_cuda(const pixel_t *nms, pixel_t *reference,
                                      const int nx, const int ny,
                                      const int tmin, int *pchanged);

__global__ void convolution_cuda(const pixel_t *in, pixel_t *out,
                                 const float *kernel, const int nx,
                                 const int ny, const int kn) {
  // index of the pixel that the thread will process
  int m = threadIdx.x + blockDim.x * blockIdx.x;
  int n = threadIdx.y + blockDim.y * blockIdx.y;

  const int khalf = kn / 2;

  // does not process pixels that are borders
  if (m >= khalf && m < nx - khalf && n >= khalf && n < ny - khalf) {
    float pixel = 0.0;
    size_t c = 0;
    for (int j = -khalf; j <= khalf; j++)
      for (int i = -khalf; i <= khalf; i++) {
        pixel += in[(n - j) * nx + m - i] * kernel[c];
        c++;
      }
    out[n * nx + m] = (pixel_t)pixel;
  }
}

__global__ void non_maximum_supression_cuda(const pixel_t *after_Gx,
                                            const pixel_t *after_Gy,
                                            const pixel_t *G, pixel_t *nms,
                                            const int nx, const int ny) {
  int m = threadIdx.x + blockDim.x * blockIdx.x;
  int n = threadIdx.y + blockDim.y * blockIdx.y;

  if (m > 0 && m < nx - 1 && n > 0 && n < ny - 1) {
    const int c = m + nx * n;
    const int nn = c - nx;
    const int ss = c + nx;
    const int ww = c + 1;
    const int ee = c - 1;
    const int nw = nn + 1;
    const int ne = nn - 1;
    const int sw = ss + 1;
    const int se = ss - 1;

    const float dir =
        (float)(fmodf(atan2f(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI) *
        8;

    if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) || // 0 deg
        ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) || // 45 deg
        ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) || // 90 deg
        ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw]))   // 135 deg
      nms[c] = G[c];
    else
      nms[c] = 0;
  }
}

__global__ void first_edges_cuda(const pixel_t *nms, pixel_t *reference,
                                 const int nx, const int ny, const int tmax) {
  int m = threadIdx.x + blockDim.x * blockIdx.x;
  int n = threadIdx.y + blockDim.y * blockIdx.y;

  size_t c = m + n * nx;
  if (m > 0 && m < nx - 1 && n > 0 && n < ny - 1) {
    if (nms[c] >= tmax) {
      reference[c] = MAX_BRIGHTNESS;
    }
  }
}

__global__ void hysteresis_edges_cuda(const pixel_t *nms, pixel_t *reference,
                                      const int nx, const int ny,
                                      const int tmin, int *pchanged) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (m > 0 && m < nx - 1 && n > 0 && n < ny - 1) {
    size_t t = m + n * nx;

    int nbs[8];
    nbs[0] = t - nx;
    nbs[1] = t + nx;
    nbs[2] = t + 1;
    nbs[3] = t - 1;
    nbs[4] = nbs[0] + 1;
    nbs[5] = nbs[0] - 1;
    nbs[6] = nbs[1] + 1;
    nbs[7] = nbs[1] - 1;

    if (nms[t] >= tmin && reference[t] == 0) {
      for (int k = 0; k < 8; k++) {
        if (reference[nbs[k]] != 0) {
          reference[t] = MAX_BRIGHTNESS;
          atomicExch(pchanged, 1); // prevent reace condition
          break;
        }
      }
    }
  }
}

__global__ void gaussian_filter_cuda(const pixel_t *in, pixel_t *out,
                                     const float *kernel, const int nx,
                                     const int ny, const int kn) {
  int m = threadIdx.x + blockDim.x * blockIdx.x;
  int n = threadIdx.y + blockDim.y * blockIdx.y;

  const int khalf = kn / 2;

  // Process pixels that are not on the border
  if (m >= khalf && m < nx - khalf && n >= khalf && n < ny - khalf) {
    float pixel = 0.0;
    int c = 0;

    for (int j = -khalf; j <= khalf; j++) {
      for (int i = -khalf; i <= khalf; i++) {
        pixel += in[(n + j) * nx + (m + i)] * kernel[c];
        c++;
      }
    }

    out[n * nx + m] = (pixel_t)pixel;
  }
}

// Helper function to generate Gaussian kernel on host
void generate_gaussian_kernel(float *kernel, int size, float sigma) {
  int half = size / 2;
  double pi = 355.0 / 113.0;
  double constant = 1.0 / (2.0 * pi * pow(sigma, 2));

  // Generate kernel values
  for (int j = -half; j <= half; j++) {
    for (int i = -half; i <= half; i++) {
      kernel[(j + half) * size + (i + half)] =
          constant * expf(-(i * i + j * j) / (2.0f * sigma * sigma));
    }
  }
}

// canny edge detector code to run on the GPU
void cannyDevice(const int *h_idata, const int w, const int h, const int tmin,
                 const int tmax, const float sigma, int *h_odata) {
  // from the class cuda exercises
  //  set up device
  int dev = 0;

  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev)); // a gpu que vou utilizar

  const int nx = w;
  const int ny = h;

  // Device memory pointers
  pixel_t *d_G, *d_after_Gx, *d_after_Gy, *d_nms;
  pixel_t *d_input, *d_output;

  // Allocate device memory
  size_t size = nx * ny * sizeof(pixel_t);
  CHECK(cudaMalloc((void **)&d_input, size));
  CHECK(cudaMalloc((void **)&d_output, size));
  CHECK(cudaMalloc((void **)&d_after_Gx, size));
  CHECK(cudaMalloc((void **)&d_after_Gy, size));
  CHECK(cudaMalloc((void **)&d_G, size));
  CHECK(cudaMalloc((void **)&d_nms, size));

  CHECK(cudaMemcpy(d_input, h_idata, size, cudaMemcpyHostToDevice));

  // test cuda devide is running

  const int blockDimX = 16;
  const int blockDimY = 16;
  // every pixel is proccess by at least one thread
  int gridDimX = (nx + blockDimX - 1) / blockDimX;
  int gridDimY = (ny + blockDimY - 1) / blockDimY;
  dim3 block(16, 16);
  dim3 grid(gridDimX, gridDimY);

  // Allocate host memory
  pixel_t *h_input = (pixel_t *)malloc(size);
  pixel_t *h_output = (pixel_t *)malloc(size);

  // Copy from device to host
  // cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);

  // Run the CPU fupppnction on host memory
  //	gaussian_filter(h_input, h_output, nx, ny, sigma);

  // Copy result back to device
  // cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);
  // Allocate a host buffer to receive the CPU’s blurred image
  const int kernel_size = 2 * (int)(2 * sigma) + 3;
  const int kn =
      (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size; // Ensure odd size
  float *h_gaussian_kernel = (float *)malloc(kn * kn * sizeof(float));
  generate_gaussian_kernel(h_gaussian_kernel, kn, sigma);

  // Copy Gaussian kernel to device
  float *d_gaussian_kernel;
  CHECK(cudaMalloc((void **)&d_gaussian_kernel, kn * kn * sizeof(float)));
  CHECK(cudaMemcpy(d_gaussian_kernel, h_gaussian_kernel,
                   kn * kn * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate temporary buffer for Gaussian output
  pixel_t *d_gaussian_output;
  CHECK(cudaMalloc((void **)&d_gaussian_output, size));

  // Apply Gaussian filter on GPU
  gaussian_filter_cuda<<<grid, block>>>(d_input, d_gaussian_output,
                                        d_gaussian_kernel, nx, ny, kn);
  CHECK(cudaDeviceSynchronize());

  // Use the Gaussian filtered output for subsequent operations
  // Copy the result to d_input for the Sobel operations
  CHECK(cudaMemcpy(d_input, d_gaussian_output, size, cudaMemcpyDeviceToDevice));

  // calculate gradiants
  const float h_Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float *d_Gx;
  CHECK(cudaMalloc((void **)&d_Gx, 9 * sizeof(float)));
  CHECK(cudaMemcpy(d_Gx, h_Gx, 9 * sizeof(float), cudaMemcpyHostToDevice));

  convolution_cuda<<<grid, block>>>(d_input, d_after_Gx, d_Gx, nx, ny, 3);
  CHECK(cudaDeviceSynchronize());

  const float h_Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  float *d_Gy;
  CHECK(cudaMalloc((void **)&d_Gy, 9 * sizeof(float)));
  CHECK(cudaMemcpy(d_Gy, h_Gy, 9 * sizeof(float), cudaMemcpyHostToDevice));
  convolution_cuda<<<grid, block>>>(d_input, d_after_Gy, d_Gy, nx, ny, 3);
  CHECK(cudaDeviceSynchronize());

  pixel_t *h_after_Gx = (pixel_t *)malloc(size);
  pixel_t *h_after_Gy = (pixel_t *)malloc(size);
  pixel_t *h_G = (pixel_t *)malloc(size);

  // copy gradients to host
  CHECK(cudaMemcpy(h_after_Gx, d_after_Gx, size, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_after_Gy, d_after_Gy, size, cudaMemcpyDeviceToHost));

  // merge gradients
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1; i < nx - 1; ++i) {
      int c = i + j * nx;
      h_G[c] = hypotf(h_after_Gx[c], h_after_Gy[c]);
    }
  }

  CHECK(cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice));

  non_maximum_supression_cuda<<<grid, block>>>(d_after_Gx, d_after_Gy, d_G,
                                               d_nms, w, h);
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemset(d_output, 0, size));

  first_edges_cuda<<<grid, block>>>(d_nms, d_output, nx, ny, tmax);
  CHECK(cudaDeviceSynchronize());

  int *d_changed;
  CHECK(cudaMalloc((void **)&d_changed, sizeof(int)));

  int h_changed;
  do {
    h_changed = 0;
    CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    hysteresis_edges_cuda<<<grid, block>>>(d_nms, d_output, nx, ny, tmin,
                                           d_changed);
    CHECK(cudaDeviceSynchronize());
    CHECK(
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
  } while (h_changed);

  CHECK(cudaMemcpy(h_odata, d_output, size, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_after_Gx);
  cudaFree(d_after_Gy);
  cudaFree(d_G);
  cudaFree(d_nms);
  cudaFree(d_Gx);
  cudaFree(d_Gy);
  cudaFree(d_changed);
  cudaFree(d_gaussian_kernel);
  cudaFree(d_gaussian_output);
  free(h_gaussian_kernel);

  free(h_after_Gx);
  free(h_after_Gy);
  free(h_G);
}
