#pragma once
#ifndef TwoDimensionalSolver_CUH
#define TwoDimensionalSolver_CUH

#include "CyclicGridsTranspose.cuh"
#include "ThomsonSolverKernel.cuh"
#include "ForwardStepKernel.cuh"
#include "CorrectionStepKernel.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion {
	template <typename T>
	struct TwoDimensionalSolver final {
		cudaError_t cycle_transpose(T **f_prev, T **f_curr, T **f_tmp, size_t x_size, size_t y_size, size_t &matrix_shift) {
			cudaError_t cudaStatus;

			float *f_prev_full = *f_prev - matrix_shift, *f_curr_full = *f_curr - matrix_shift, *f_tmp_full = *f_tmp - matrix_shift;
			if (cudaSuccess != (cudaStatus = iki::diffusion::cyclic_grids_transpose<tile_dim, block_rows>(f_prev_full, f_curr_full, f_tmp_full, x_size, y_size))) {
				return cudaStatus;
			}

			auto rotate_tmp = f_prev_full;
			f_prev_full = f_curr_full;
			f_curr_full = f_tmp_full;
			f_tmp_full = rotate_tmp;

			matrix_shift = x_size + 1;
			*f_prev = f_prev_full + matrix_shift;
			*f_curr = f_curr_full + matrix_shift;
			*f_tmp = f_tmp_full + matrix_shift;

			return cudaStatus;
		}

		cudaError_t step(T **f_prev, T **f_curr, T **f_tmp, T *x_dfc, T *y_dfc, T *a, T *b, T *c, T *d, T rx, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int threads, blockDim;
			size_t matrix_shift = x_size + 1;

			blockDim = 1; threads = x_size - 2;
			device::forward_step_multisolver_kernel << <blockDim, threads >> > (*f_prev, x_dfc, y_dfc, a, b, c, d, rx, ry, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, *f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			if (cudaSuccess != (cudaStatus = cycle_transpose<32u, 8u>(f_prev, f_curr, f_tmp, x_size, y_size, matrix_shift)))
				return cudaStatus;

			blockDim = 1; threads = y_size - 2;
			device::correction_step_multisolver_kernel<<<blockDim, threads>>>(*f_prev, *f_curr, y_dfc, a, b, c, d, ry, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, *f_curr, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			if (cudaSuccess != (cudaStatus = cycle_transpose<32u, 8u>(f_prev, f_curr, f_tmp, y_size, x_size, matrix_shift)))
				return cudaStatus;

			std::swap(*f_prev, *f_curr);
			return cudaStatus;
		}
	};
} /* diffusion */ } /* iki */

#endif /* TwoDimensionalSolver_CUH */