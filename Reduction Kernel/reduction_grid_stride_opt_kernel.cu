#include <stdio.h>

#include "reduction.cuh"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ void grid_stride_opt_reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    // Calculate input with grid-stride loop and save to shared memory
    float input[NUM_LOAD] = { 0.f };
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
        {
            input[step] += (i + step * blockDim.x * gridDim.x < size) ? d_in[i + step * blockDim.x * gridDim.x] : 0.f;
        }
    }
    for (int i = 1; i < NUM_LOAD; i++)
    {
        input[0] += input[i];
    }
    s_data[threadIdx.x] = input[0];

    __syncthreads();

    // Perform reduction with sequential addressing
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = s_data[0];
    }
}

void grid_stride_opt_reduction(float* d_out, float* d_in, int n_threads, unsigned int size)
{
    int num_sms;
    int num_blocks_per_sm;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, grid_stride_opt_reduction_kernel, n_threads, n_threads * sizeof(float));

    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    grid_stride_opt_reduction_kernel << <n_blocks, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, size);
    grid_stride_opt_reduction_kernel << <1, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, n_blocks);
}