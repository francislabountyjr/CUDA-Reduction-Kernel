#include <stdio.h>

#include "reduction.cuh"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

using namespace cooperative_groups;

// cuda thread synchronization
__global__ void cooperative_group_reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    extern __shared__ float s_data[];

    // Calculate input with grid-stride loop and save to shared memory
    float input[NUM_LOAD] = { 0.f };
    for (int i = idx_x; i < size; i += block.group_dim().x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
        {
            input[step] += (i + step * block.group_dim().x * gridDim.x < size) ? d_in[i + step * block.group_dim().x * gridDim.x] : 0.f;
        }
    }
    for (int i = 1; i < NUM_LOAD; i++)
    {
        input[0] += input[i];
    }
    s_data[threadIdx.x] = input[0];

    block.sync();

    // Perform reduction
    for (unsigned int stride = block.group_dim().x / 2; stride > 0; stride >>= 1)
    {
        // Scheduled threads reduce for every iteration
        // Will eventually be smaller than a warp size (32)
        if (block.thread_index().x < stride)
        {
            s_data[block.thread_index().x] += s_data[block.thread_index().x + stride];
        }

        block.sync();
    }

    if (block.thread_index().x == 0)
    {
        d_out[block.group_index().x] = s_data[0];
    }
}

void cooperative_group_reduction(float* d_out, float* d_in, int n_threads, unsigned int size)
{
    int num_sms;
    int num_blocks_per_sm;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, cooperative_group_reduction_kernel, n_threads, n_threads * sizeof(float));

    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    cooperative_group_reduction_kernel << <n_blocks, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, size);
    cooperative_group_reduction_kernel << <1, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, n_blocks);
}