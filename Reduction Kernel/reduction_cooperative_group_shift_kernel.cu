#include <stdio.h>

#include "reduction.cuh"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

using namespace cooperative_groups;

template <typename group_t> __inline__ __device__ float warp_reduce_sum(group_t group, float val)
{
    for (int offset = group.size() / 2; offset > 0; offset >>= 1)
    {
        val += group.shfl_down(val, offset);
    }

    return val;
}

__inline__ __device__ float block_reduce_sum(thread_block block, float val)
{
    __shared__ float shared[32]; // Shared memory for 32 partial sums
    int warp_idx = block.thread_index().x / warpSize;

    // partial reduction at tile<32> size
    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    val = warp_reduce_sum(tile32, val);

    // Write reduced value to shared memory
    if (tile32.thread_rank() == 0)
    {
        shared[warp_idx] = val;
    }

    block.sync(); // Wait for all partial reductions

    // Read from shared memory if leading warp
    if (warp_idx == 0)
    {
        val = (threadIdx.x < block.group_dim().x / warpSize) ? shared[tile32.thread_rank()] : 0;
        // Final reduce within the first warp
        val = warp_reduce_sum(tile32, val);
    }

    return val;
}

// cuda thread synchronization
__global__ void cooperative_group_shift_reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    // Calculate input with grid-stride loop and save to shared memory
    float sum[NUM_LOAD] = { 0.f };
    for (int i = idx_x; i < size; i += block.group_dim().x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
        {
            sum[step] += (i + step * block.group_dim().x * gridDim.x < size) ? d_in[i + step * block.group_dim().x * gridDim.x] : 0.f;
        }
    }
    for (int i = 1; i < NUM_LOAD; i++)
    {
        sum[0] += sum[i];
    }
    
    // Warp synchronous reduction
    sum[0] = block_reduce_sum(block, sum[0]);

    if (block.thread_index().x == 0)
    {
        d_out[block.group_index().x] = sum[0];
    }
}

void cooperative_group_shift_reduction(float* d_out, float* d_in, int n_threads, unsigned int size)
{
    int num_sms;
    int num_blocks_per_sm;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, cooperative_group_shift_reduction_kernel, n_threads, n_threads * sizeof(float));

    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    cooperative_group_shift_reduction_kernel << <n_blocks, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, size);
    cooperative_group_shift_reduction_kernel << <1, n_threads, n_threads * sizeof(float), 0 >> > (d_out, d_out, n_blocks);
}