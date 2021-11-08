#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

// @reduction_global_kernel.cu
void global_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_shared_kernel.cu
void shared_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_interleaving_kernel.cu
void interleaving_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_sequential_kernel.cu
void sequential_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_grid_stride_kernel.cu
void grid_stride_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_grid_stride_opt_kernel.cu
void grid_stride_opt_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_warp_primitive_kernel.cu
void warp_primitive_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_cooperative_group_kernel.cu
void cooperative_group_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_cooperative_group_shift_kernel.cu
void cooperative_group_shift_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @reduction_cooperative_group_shift_unrolled_kernel.cu
void cooperative_group_shift_unrolled_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

// @atomic_kernel.cu
void atomic_reduction(float* d_out, float* d_in, int n_threads, unsigned int size);

#define max(a, b) (a) > (b) ? (a) : (b)

#define min(a, b) (a) < (b) ? (a) : (b)

#define NUM_LOAD 4