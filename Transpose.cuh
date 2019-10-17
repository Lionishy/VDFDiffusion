#pragma once
#ifndef Transpose_CUH
#define Transpose_CUH

#include <cuda_runtime.h>

template <typename T>
__global__ void naiive_transpose_kernel(T *dst, T const *src, size_t x_size, size_t y_size, size_t tile_size) {
	size_t index_in = blockIdx.x * tile_size + blockIdx.y * tile_size * x_size;
	size_t index_out = blockIdx.y * tile_size + blockIdx.x * tile_size * y_size;
	dst[index_out + threadIdx.y + threadIdx.x * y_size] = src[index_in + threadIdx.x + threadIdx.y * x_size];
}

template <typename T>
__global__ void simple_transpose_kernel(T *dst, T const *src, size_t x_size, size_t y_size, size_t tile_size) {
	__shared__ float tile[]; //dynamically allocated memory
	size_t index_in = blockIdx.x * tile_size + blockIdx.y * tile_size *x_size;
	tile[threadIdx.y * tile_size + threadIdx.x] = src[index_in + threadIdx.x + threadIdx.y*x_size];
	
	__syncthreads();

	size_t index_out = blockIdx.y * tile_size + blockIdx.x * tile_size * y_size;
	dst[index_out + threadIdx.x + threadIdx.y * y_size] = tile[threadIdx.x * tile_size + thread.y];
}

#endif /* Transpose_CUH */