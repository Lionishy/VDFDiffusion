#pragma once
#ifndef Transpose_CUH
#define Transpose_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <unsigned TILE_DIM, unsigned BLOCK_ROWS, typename T>
__global__ void transpose_kernell(T *dst, T const *src, unsigned x_size, unsigned y_size) {
	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	unsigned xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	unsigned yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	unsigned index_in = xIndex + yIndex * x_size;


	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	unsigned index_out = xIndex + yIndex * y_size;


	for (unsigned i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		tile[threadIdx.y + i][threadIdx.x] =
			src[index_in + i * x_size];
	}

	__syncthreads();

	for (unsigned i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		dst[index_out + i * y_size] =
			tile[threadIdx.x][threadIdx.y + i];
	}
}

#endif /* Transpose_CUH */