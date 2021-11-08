#include <stdio.h>

#include "reduction.cuh"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ void global_reduction_kernel(float* d_out, float* d_in, unsigned int stride, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_x + stride < size)
    {
        d_out[idx_x] += d_in[idx_x + stride];
    }
}

void global_reduction(float* d_out, float* d_in, int n_threads, unsigned int size)
{
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int n_blocks = (size + n_threads - 1) / n_threads;

    for (unsigned int stride = 1; stride < size; stride *= 2)
    {
        global_reduction_kernel<<<n_blocks,n_threads>>>(d_out, d_out, stride, size);
    }
}