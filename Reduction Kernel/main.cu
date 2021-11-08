#include <stdio.h>
#include <stdlib.h>
#include <helper_timer.h>

#include "reduction.cuh"

void run_benchmark(void (*reduce)(float*, float*, int, unsigned int), float* d_outPtr, float* d_inPtr, unsigned int size);

void init_input(float* data, unsigned int size);

float get_cpu_result(float* data, unsigned int size);

int main(int argc, char* argv[])
{
	float* h_inPtr;
	float* d_inPtr, * d_outPtr;

	unsigned int size = 1 << 24;

	float result_host, result_gpu;

	srand(2021);

	// Allocate memory

	h_inPtr = (float*)malloc(size * sizeof(float));

	// Data initialization with random values
	init_input(h_inPtr, size);

	// Prepare GPU resources
	cudaMalloc((void**)&d_inPtr, size * sizeof(float));
	cudaMalloc((void**)&d_outPtr, size * sizeof(float));

	cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

	// Get reduction result from GPU
	printf("-------Global-------\n");
	run_benchmark(global_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Shared-------\n");
	run_benchmark(shared_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Interleaving-------\n");
	run_benchmark(interleaving_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Sequential-------\n");
	run_benchmark(sequential_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Grid Stride-------\n");
	run_benchmark(grid_stride_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Grid Stride: Optimized-------\n");
	run_benchmark(grid_stride_opt_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Warp Primitive-------\n");
	run_benchmark(warp_primitive_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Cooperative Groups-------\n");
	run_benchmark(cooperative_group_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Cooperative Groups: Shift-------\n");
	run_benchmark(cooperative_group_shift_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Cooperative Groups: Shift Loop Unrolled-------\n");
	run_benchmark(cooperative_group_shift_unrolled_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	printf("-------Atomic: Block Level Add-------\n");
	run_benchmark(atomic_reduction, d_outPtr, d_inPtr, size);
	cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);
	printf("Device: %f\n", result_gpu);

	// Get reduction result from CPU
	result_host = get_cpu_result(h_inPtr, size);
	printf("Host: %f\n", result_host);

	// Terminate memory
	cudaFree(d_outPtr);
	cudaFree(d_inPtr);
	free(h_inPtr);

	return 0;
}

void run_benchmark(void(*reduce)(float*, float*, int, unsigned int), float* d_outPtr, float* d_inPtr, unsigned int size)
{
	int num_threads = 256;
	int test_iter = 100;

	// Warm-up
	reduce(d_outPtr, d_inPtr, num_threads, size);

	// Initialize timer
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// Conduct test iterations
	for (int i = 0; i < test_iter; i++)
	{
		reduce(d_outPtr, d_inPtr, num_threads, size);
	}

	// Get elapsed time and stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);

	// Compute and print the performance
	float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
	float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6;
	printf("Time: %.3f msec\nBandwidth: %f GB/s\n", elapsed_time_msed, bandwidth);

	// Delete the timer
	sdkDeleteTimer(&timer);
}

void init_input(float* data, unsigned int size)
{
	for (int i = 0; i < size; i++)
	{
		// Keep numbers small to avoid truncation error in the sum
		data[i] = (rand() & 0xFF) / (float)RAND_MAX;
	}
}

float get_cpu_result(float* data, unsigned int size)
{
	double result = 0.f;
	for (int i = 0; i < size; i++)
	{
		result += data[i];
	}

	return (float)result;
}
