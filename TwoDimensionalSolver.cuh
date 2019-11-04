#pragma once
#ifndef TwoDimensionalSolver_CUH
#define TwoDimensionalSolver_CUH

#include "CyclicGridsTranspose.cuh"
#include "ThomsonSolverKernel.cuh"
#include "ForwardStepKernel.cuh"
#include "CorrectionStepKernel.cuh"
#include "ForwardStepCorrectionKernel.cuh"

#include <cuda_runtime.h>

namespace iki { namespace diffusion {
	template <typename T>
	struct TwoDimensionalSolver final {
		TwoDimensionalSolver(size_t x_size, size_t y_size, T rx, T ry, T rxy, T *f_prev_full, T *f_curr_full, T *f_tmp_full, T *x_dfc, T *y_dfc, T *xy_dfc, T *yx_dfc, T *a, T *b, T *c, T *d): x_size(x_size), y_size(y_size), rx(rx), ry(ry), rxy(rxy), f_prev_full(f_prev_full), f_curr_full(f_curr_full), f_tmp_full(f_tmp_full), x_dfc(x_dfc), y_dfc(y_dfc), xy_dfc(xy_dfc), yx_dfc(yx_dfc), a(a), b(b), c(c), d(d) {
			f_prev = f_prev_full + y_size + 1;
			f_curr = f_curr_full + y_size + 1;
			f_tmp = f_tmp_full + y_size + 1;
		}

		cudaError_t cycle_transpose(size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = iki::diffusion::cyclic_grids_transpose<32u, 8u>(f_prev_full, f_curr_full, f_tmp_full, x_size, y_size))) {
				return cudaStatus;
			}

			auto rotate_tmp = f_prev_full;
			f_prev_full = f_curr_full;
			f_curr_full = f_tmp_full;
			f_tmp_full = rotate_tmp;

			f_prev = f_prev_full + x_size + 1;
			f_curr = f_curr_full + x_size + 1;
			f_tmp = f_tmp_full + x_size + 1;

			return cudaStatus;
		}

		cudaError_t forward_step(T *x_dfc, T *y_dfc, T *xy_dfc, T *yx_dfc, T rx, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			return cudaStatus;
		}

		cudaError_t forward_step_with_mixed_terms_correction(T *x_dfc, T *y_dfc, T *xy_dfc, T *yx_dfc, T rx, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			device::forward_step_correction_multisolver_kernel<<<blockDim, threads>>>(f_prev, f_curr, xy_dfc, yx_dfc, d, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			math::device::thomson_multisolver_kernell << <blockDim, threads >> > (a, b, c, d, f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			return cudaStatus;
		}

		cudaError_t correction_step(T *y_dfc, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2; 

			device::correction_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, f_curr, y_dfc, a, b, c, d, ry, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				return cudaStatus;

			return cudaStatus;
		}

		cudaError_t step() {
			cudaError_t cudaStatus;
			//f_prev -> f_curr
			if (cudaSuccess != (cudaStatus = forward_step(x_dfc, y_dfc, xy_dfc, yx_dfc, rx, ry, x_size, y_size)))
				return cudaStatus;
			
			if (cudaSuccess != (cudaStatus = cycle_transpose(x_size, y_size)))
				return cudaStatus;
			
			//f_prev,f_curr -> f_curr
			if (cudaSuccess != (cudaStatus = correction_step(y_dfc, ry, x_size, y_size)))
				return cudaStatus;

			if (cudaSuccess != (cudaStatus = cycle_transpose(y_size, x_size)))
				return cudaStatus;

			//f_prev,f_curr -> f_curr
			if (cudaSuccess != (cudaStatus = forward_step_with_mixed_terms_correction(x_dfc, y_dfc, xy_dfc, yx_dfc, rx, ry, x_size, y_size)))
				return cudaStatus;

			if (cudaSuccess != (cudaStatus = cycle_transpose(x_size, y_size)))
				return cudaStatus;

			//f_prev,f_curr -> f_curr
			if (cudaSuccess != (cudaStatus = correction_step(y_dfc, ry, x_size, y_size)))
				return cudaStatus;

			if (cudaSuccess != (cudaStatus = cycle_transpose(y_size, x_size)))
				return cudaStatus;

			std::swap(f_prev_full, f_curr_full);
			std::swap(f_prev, f_curr);
			return cudaStatus;
		}

		size_t const x_size, y_size;
		T const rx, ry, rxy;
		T *f_prev_full, *f_curr_full, *f_tmp_full;
		T *f_prev, *f_curr, *f_tmp;
		T *x_dfc, *y_dfc, *xy_dfc, *yx_dfc;
		T *a, *b, *c, *d;
	};
} /* diffusion */ } /* iki */

#endif /* TwoDimensionalSolver_CUH */