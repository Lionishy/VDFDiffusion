#pragma once
#ifndef TwoDimensionalSolver_CUH
#define TwoDimensionalSolver_CUH

#include "CyclicGridsTranspose.cuh"
#include "ThomsonSolverKernel.cuh"
#include "ForwardStepKernel.cuh"
#include "CorrectionStepKernel.cuh"
#include "ForwardStepCorrectionKernel.cuh"
#include "DeviceMemory.h"
#include "DeviceException.h"

#include <cuda_runtime.h>
#include <vector>

namespace iki { namespace diffusion {
	template <typename T>
	struct AbstractTwoDimensionalSolver {
		AbstractTwoDimensionalSolver(size_t x_size, size_t y_size, T rx, T ry, std::vector<T> const &f, std::vector<T> const &x_dfc_host, std::vector<T> const &y_dfc_host, std::vector<T> const &xy_dfc_host, std::vector<T> const &yx_dfc_host): device_ptr(11 * x_size * y_size * sizeof(T)), x_size(x_size), y_size(y_size), rx(rx), ry(ry), rxy(std::sqrt(rx * ry)) {
			//pointers assignment
			{
				size_t matrix_size = x_size * y_size;
				f_prev_full = (T*)device_ptr.get();
				f_curr_full = f_prev_full + matrix_size;
				f_tmp_full = f_curr_full + matrix_size;
				x_dfc = f_tmp_full + matrix_size;
				y_dfc = x_dfc + matrix_size;
				xy_dfc = y_dfc + matrix_size;
				yx_dfc = xy_dfc + matrix_size;
				a = yx_dfc + matrix_size;
				b = a + matrix_size;
				c = b + matrix_size;
				d = c + matrix_size;
			}

			//initial data copy
			{
				cudaError_t cudaStatus;
				if (cudaSuccess != (cudaStatus = cudaMemcpy(f_prev_full, f.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(f_curr_full, f.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(x_dfc, x_dfc_host.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(y_dfc, y_dfc_host.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(xy_dfc, xy_dfc_host.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(yx_dfc, yx_dfc_host.data(), x_size * y_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
			}
			
			//main grid pointers shift
			f_prev = f_prev_full + y_size + 1, f_curr = f_curr_full + y_size + 1, f_tmp = f_tmp_full + y_size + 1;
			x_dfc += y_size + 1, y_dfc += x_size + 1, xy_dfc += y_size + 1, yx_dfc += x_size + 1;
		}

		//precaution to avoid unintended usage of copy / assign / move
		AbstractTwoDimensionalSolver(AbstractTwoDimensionalSolver const &src) = delete;
		AbstractTwoDimensionalSolver &operator=(AbstractTwoDimensionalSolver const &src) = delete;
		AbstractTwoDimensionalSolver(AbstractTwoDimensionalSolver &&src) = delete;
		AbstractTwoDimensionalSolver &operator=(AbstractTwoDimensionalSolver &&src) = delete;

		void cycle_transpose(size_t x_size, size_t y_size) {
			{
				cudaError_t cudaStatus;
				if (cudaSuccess != (cudaStatus = iki::diffusion::cyclic_grids_transpose<32u, 8u>(f_prev_full, f_curr_full, f_tmp_full, x_size, y_size))) throw DeviceException(cudaStatus);
			}

			auto rotate_tmp = f_prev_full;
			f_prev_full = f_curr_full;
			f_curr_full = f_tmp_full;
			f_tmp_full = rotate_tmp;

			f_prev = f_prev_full + x_size + 1;
			f_curr = f_curr_full + x_size + 1;
			f_tmp = f_tmp_full + x_size + 1;
		}

		void forward_step(T *x_dfc, T *y_dfc, T *xy_dfc, T *yx_dfc, T rx, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);
		}

		void forward_step_with_mixed_terms_correction(T *x_dfc, T *y_dfc, T *xy_dfc, T *yx_dfc, T rx, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, x_dfc, y_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);

			device::forward_step_correction_multisolver_kernel<<<blockDim, threads>>>(f_prev, f_curr, xy_dfc, yx_dfc, d, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);
		}

		void correction_step(T *y_dfc, T ry, size_t x_size, size_t y_size) {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2; 

			device::correction_step_multisolver_kernel<<<blockDim, threads>>>(f_prev, f_curr, y_dfc, a, b, c, d, ry, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError())) throw DeviceException(cudaStatus);
		}

		virtual void step() = 0;

		DeviceMemory device_ptr;
		size_t const x_size, y_size;
		T const rx, ry, rxy;
		T *f_prev_full, *f_curr_full, *f_tmp_full;
		T *f_prev, *f_curr, *f_tmp;
		T *x_dfc, *y_dfc, *xy_dfc, *yx_dfc;
		T *a, *b, *c, *d;
	};
} /* diffusion */ } /* iki */

#endif /* TwoDimensionalSolver_CUH */