# cle-assignment3

### guassian_filter()

The provided _guassian\_filter_ function calculates the 1D kernel, using guassian filter formula.

To apply to CUDA, first the kernel is generated on the host, this time using the 2D formula, and then applied in chunks by each thread.

This is then followed by waiting for the thread synchronization with cudaDeviceSynchronize.


### convolution()

The provided _convolution_ function performs a 2D convolution operation on an input image, using a given kernel of width kn to produce an outpuy image. It iterates through each pixel of the image (excluding borders. For each pixel, it computes a weighted sum of its neighbors based on the kernel values. This involves two inner nested loops for the kernel elements.

Using a CUDA kernel, each thread will parallelize the convulotion, assigning each thread to compute the output for a single pixel. So, first it calculates the index of the pixel, using the `threadIdx`, `blockDim` and `blockIdx`. Then check if the pixel is not a boudrie of the image, because it need to be ignore. Then the inner loop is the same as the other implementation.

The convolutino callculates gradients using a kernel, so, it is necessary to allocate memory in the device and copy the kernel from the host (`h_Gx`) to the device kernel (`g_Gx`), using `cudaMalloc`and `cudaMemcpy`, in order. After the computation, we use `cudaDeviceSynchronize` to wait for all the threads to finish. And repeat the process for the vertical gradient.

Finnaly, we copy the horizontal and vertical gradients from the device to the host (`cudaMemcpy` with `cudaMemcpyDeviceToHost`) to performe the merge of the gradients in the cpu. And then copy the result (`h_G`) to the device (`d_G`).

### non\_maximum\_suppression()

The provided _non maximum suppression_ function refines the detected edges. For each pixel, it examines the gradient magnitude and direction. It compares the current pixel's gradient magnitude with its two neighbors along the gradient direction. If the current pixel's magnitude is not greater than both neighbors, it is suppressed (set to 0), otherwise, it is kept. This involves iterating through each pixel (excluding borders).

Similar to the convolution, the CUDA Kernel implementation assign to each thread one pixel, using the `threadIdx`, `blockDim` and `blockIdx` to calculate the index of the pixel assign. Then, performe the same logic as the function provided uses inside the loop.

The function has the horizontal and vertical gradients has input, as well has the computed merged gradient, that, firstly, it needs to be copied from the host to the device. Producing the output `d_nms`. (In the beggining of the main funtion, there is space allocated to `d_nms`, `cudaMalloc((void **)&d_nms, size)`, being the size, the size of the image)

### first\_edges()

The provided _first\_edges_ function performs the initial part of the double thresholding step. It iterates through the non-maximum suppressed image (nms). If a pixel's value in nms is greater than or equal to a high threshold (tmax), it is marked as a strong edge in the reference image by setting its value to MAX_BRIGHTNESS

To call the function, first it is needed to set the output variable in the device (`cudaMemset(d\_output, 0, size)`, where, previous, memory was allocated). As for the input, it is used the output of the non maximum supression (`d_nms`) that is already on the device.

### hysteresis\_edges()

The provided _hysteresis\_edges_ function performs the edge tracking part of Canny. It iterates through pixels. If a pixel's nms value is above a low threshold (tmin) and it is not yet marked as an edge (reference[t] == 0), it checks its 8 neighbors. If any neighbor is already marked as an edge (reference[nbs[k]] != 0), the current pixel is also marked as an edge (reference[t] = MAX_BRIGHTNESS), and a flag *pchanged is set to true. This function is called repeatedly until no more changes occur.

To adapt to use CUDA, the difference is in the use of the `atomicExch(pchanged, 1)`, that will signal if any thread made a change. this operation writes a value to a memory location and returns the old value at that location, all in one indivisible operation. This prevents race conditions where multiple threads might try to set *pchanged to true simultaneously. It is allocated memory in the device for the variable `d_changed`, which is an integer, that will work as boolean, 1 being true and 0 false. And the `pChanged` will write to this space, and a copy to the host correspondent variable is made (to `h_changed`). It is performed a do-while loop that breaks when no more changes appens, when `d_changed\h_changed` is 0.

