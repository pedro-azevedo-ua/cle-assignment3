# cle-assignment3

### convolution()

The provided _convolution_ function performs a 2D convolution operation on an input image, using a given kernel of width kn to produce an outpuy image. It iterates through each pixel of the image (excluding borders. For each pixel, it computes a weighted sum of its neighbors based on the kernel values. This involves two inner nested loops for the kernel elements. 

Using a CUDA kernel, each thread will parallelize the convulotion, assigning each thread to compute the output for a single pixel. So, first it calculates the index of the pixel, using the `threadIdx`, `blockDim` and `blockIdx`. Then check if the pixel is not a boudrie of the image, because it need to be ignore. Then the inner loop is the same as the other implementation.

The convolutino callculates gradients using a kernel, so, it is necessary to allocate memory in the device and copy the kernel from the host (`h_Gx`) to the device kernel (`g_Gx`), using `cudaMalloc`and `cudaMemcpy`, in order. After the computation, we use `cudaDeviceSynchronize` to wait for all the threads to finish. And repeat the process for the vertical gradient.

Finnaly, we copy the horizontal and vertical gradients from the device to the host (`cudaMemcpy` with `cudaMemcpyDeviceToHost`) to performe the merge of the gradients in the cpu. And then copy the result (`h_G`) to the device (`d_G`).

### non\_maximum\_suppression()

The provided _non maximum suppression_ function refines the detected edges. For each pixel, it examines the gradient magnitude and direction. It compares the current pixel's gradient magnitude with its two neighbors along the gradient direction. If the current pixel's magnitude is not greater than both neighbors, it is suppressed (set to 0), otherwise, it is kept. This involves iterating through each pixel (excluding borders).

Similar to the convolution, the CUDA Kernel implementation assign to each thread one pixel, using the `threadIdx`, `blockDim` and `blockIdx` to calculate the index of the pixel assign. Then, performe the same logic as the function provided uses inside the loop.

The function has the horizontal and vertical gradients has input, as well has the computed merged gradient, that, firstly, it needs to be copied from the host to the device. Producing the output `d_nms`. (In the beggining of the main funtion, there is space allocated to `d_nms`, `cudaMalloc((void **)&d_nms, size)`, being the size, the size of the image)
