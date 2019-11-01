#pragma once
#ifndef CyclicGridsTranspose_CUH
#define CyclicGridsTranspose_CUH

#include "Transpose.cuh"
#include <cuda_runtime.h>

namespace iki { namespace diffusion { 
	template <unsigned tile_dim, unsigned block_rows, typename T>
	cudaError_t cyclic_grids_transpose(T *f_prev, T *f_curr, T *f_tmp, size_t x_size, size_t y_size) {
		cudaError_t cudaStatus;

		dim3 grid(x_size / tile_dim, y_size / tile_dim), threads(tile_dim, block_rows);
		iki::math::device::transpose_kernell<tile_dim, block_rows><<<grid, threads>>>(f_tmp, f_curr, x_size, y_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			return cudaStatus;

		iki::math::device::transpose_kernell<tile_dim, block_rows><<<grid, threads>>>(f_curr, f_prev, x_size, y_size);
		if (cudaSuccess != (cudaStatus = cudaGetLastError()))
			return cudaStatus;

		return cudaStatus;
	}
} /* diffusion */ } /* iki */

#endif /* CyclicGridsTranspose_CUH */