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
	__shared__ float tile[1024]; //dynamically allocated memory
	size_t index_in = blockIdx.x * tile_size + blockIdx.y * tile_size *x_size;
	tile[threadIdx.y * tile_size + threadIdx.x] = src[index_in + threadIdx.x + threadIdx.y*x_size];
	
	__syncthreads();

	size_t index_out = blockIdx.y * tile_size + blockIdx.x * tile_size * y_size;
	dst[index_out + threadIdx.x + threadIdx.y * y_size] = tile[threadIdx.x * tile_size + threadIdx.y];
}

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void paper_transpose_kernell(float *odata, float const *idata, size_t width, size_t height) {
	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int index = xIndex + width * yIndex;
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		tile[threadIdx.y + i][threadIdx.x] =
			idata[index + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		odata[index + i * width] =
			tile[threadIdx.y + i][threadIdx.x];
	}
}

__global__ void paper_diagonal_transpose_kernell(float *odata, float const *idata, size_t width, size_t height) {
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int blockIdx_x, blockIdx_y;
	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}
	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		tile[threadIdx.y + i][threadIdx.x] =
			idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		odata[index_out + i * height] =
			tile[threadIdx.x][threadIdx.y + i];
	}
}

#endif /* Transpose_CUH */