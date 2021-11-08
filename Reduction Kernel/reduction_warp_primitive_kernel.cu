#include <stdio.h>

#include "reduction.cuh"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        unsigned int mask = __activemask();
        val += __shfl_down_sync(mask, val, offset);
    }

    return val;
}

__inline__ __device__ float block_reduce_sum(float val)
{
    static __shared__ float shared[32]; // Shared memory for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    val = warp_reduce_sum(val);

    if (lane == 0)
    {
        shared[wid] = val; // Write reduced value to shared memory
    }

    __syncthreads(); // Wait for all partial reductions

    // Read from shared memory if that warp existed
    if (wid == 0)
    {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        val = warp_reduce_sum(val); // Final reduce within the first warp
    }

    return val;
}

__global__ void warp_primitive_reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Cumulate input with grid-stride loop and save to shared memory
    float sum[NUM_LOAD] = { 0.f };
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
        {
            sum[step] += (i + step * blockDim.x * gridDim.x < size) ? d_in[i + step * blockDim.x * gridDim.x] : 0.f;
        }
    }
    for (int i = 1; i < NUM_LOAD; i++)
    {
        sum[0] += sum[i];
    }
    
    sum[0] = block_reduce_sum(sum[0]);

    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = sum[0];
    }
    /*float sum = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x)
    {
        sum += d_in[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = sum;
    }*/
}

void warp_primitive_reduction(float* d_out, float* d_in, int n_threads, unsigned int size)
{
    int num_sms;
    int num_blocks_per_sm;

    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, warp_primitive_reduction_kernel, n_threads, n_threads * sizeof(float));

    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    warp_primitive_reduction_kernel<<<n_blocks,n_threads,n_threads*sizeof(float),0>>>(d_out, d_out, size);
    warp_primitive_reduction_kernel<<<1,n_threads,n_threads*sizeof(float),0>>>(d_out, d_out, n_blocks);
}