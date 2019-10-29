#pragma once
#ifndef TwoDimensionalDeviceSolver_CUH
#define TwoDimensionalDeviceSolver_CUH

#include "CyclicGridsTranspose.cuh"
#include "ForwardStepKernel.cuh"
#include "ForwardStepCorrectionKernel.cuh"
#include "CorrectionStepKernel.cuh"
#include "ThomsonSolverKernel.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace iki { namespace diffusion {
	template <typename T>
	struct TwoDimensionalSolver final {
		static std::runtime_error cuda_error_construct(cudaError_t cudaStatus, std::string const &action) {
			std::stringstream error_stream;
			error_stream << action << " failed!\n" << "Device code: " << cudaStatus << "\n" << "Device string: " << cudaGetErrorString(cudaStatus);
			return std::runtime_error(error_stream.str());
		}

		static void from_host_to_device(T *dst_dev, T const *src_host, size_t bytes) {
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice)))
				throw cuda_error_construct(cudaStatus,"Copying data from host to device");
		}

		static void from_device_to_host(T *dst_host, T const *src_dev, size_t bytes) {
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost)))
				throw cuda_error_construct(cudaStatus, "Copying data from host to device");
		}

		TwoDimensionalSolver(std::ostream &debug_stream, size_t x_size, size_t y_size, T rx, T ry, T rxy): debug_stream(debug_stream), x_size(x_size), y_size(y_size), rx(rx), ry(ry), rxy(rxy), matrix_size(x_size*y_size), global_device_memory(NULL) {
			size_t bytes = 3 * x_size * y_size * sizeof(T) + 4 * x_size * y_size * sizeof(T) + 4 * x_size * y_size * sizeof(T);
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMalloc(&global_device_memory, bytes)))	{
				std::stringstream error_stream;
				error_stream << "Allocation of " << ((bytes - 1) / 1024 + 1) << "Kb on device";
				throw cuda_error_construct(cudaStatus, error_stream.str());
			}
			else {
				debug_stream << "Successfully allocated " << (bytes - 1) / 1024 + 1 << "Kb on device" << std::endl;
			}

			f_prev = (T*)global_device_memory;
			f_curr = f_prev + matrix_size;
			f_tmp = f_curr + matrix_size;
			dfc = f_tmp + matrix_size;
			xx_dfc = dfc + y_size + 1;
			yy_dfc = dfc + matrix_size + x_size + 1;
			xy_dfc = dfc + 2 * matrix_size + y_size + 1;
			yx_dfc = dfc + 3 * matrix_size + x_size + 1;
			m = dfc + 4 * matrix_size;
			
			grid_pointers_calculation(x_size + 1);
		}

		void init(T const *f_host, T const *xx_dfc_host, T const *yy_dfc_host, T const *xy_dfc_host, T const *yx_dfc_host) {
			from_host_to_device(f_prev, f_host, matrix_size * sizeof(T));
			from_host_to_device(f_curr, f_host, matrix_size * sizeof(T));
			from_host_to_device(dfc, xx_dfc_host, matrix_size * sizeof(T));
			from_host_to_device(dfc + matrix_size, yy_dfc_host, matrix_size * sizeof(T));
			from_host_to_device(dfc + 2 * matrix_size, xy_dfc_host, matrix_size * sizeof(T));
			from_host_to_device(dfc + 3 * matrix_size, yx_dfc_host, matrix_size * sizeof(T));
		}

		void retrieve(T *f_host) {
			from_device_to_host(f_host, f_prev, matrix_size * sizeof(T));
		}

		void forward_step() {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev_grid, xx_dfc, yy_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Forward step matrix calculation");

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr_grid, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Forward step Thomson solver");
		}

		void forward_step_with_mixed_terms_correction() {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = x_size - 2;

			device::forward_step_multisolver_kernel<<<blockDim, threads>>>(f_prev_grid, xx_dfc, yy_dfc, xy_dfc, yx_dfc, a, b, c, d, rx, ry, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Forward step with mixed terms correction matrix calculation");

			device::forward_step_correction_multisolver_kernel<<<blockDim, threads>>>(f_prev_grid, f_curr_grid, xy_dfc, yx_dfc, d, rxy, x_size - 2, y_size, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Forward step with mixed terms correction free terms correction");

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr_grid, x_size - 2, y_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Forward step with mixed terms correction Thomson solver");
		}

		void correction_step() {
			cudaError_t cudaStatus;
			int blockDim = 1, threads = y_size - 2;

			device::correction_step_multisolver_kernel<<<blockDim, threads>>>(f_prev_grid, f_curr_grid, yy_dfc, a, b, c, d, ry, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Correction step matrix calculation");

			math::device::thomson_multisolver_kernell<<<blockDim, threads>>>(a, b, c, d, f_curr_grid, y_size - 2, x_size);
			if (cudaSuccess != (cudaStatus = cudaGetLastError()))
				throw cuda_error_construct(cudaStatus, "Correction step Thomson solver");
		}

		void x_to_y_transpose(std::string const &action) {
			cudaError_t cudaStatus;

			if (cudaSuccess != (cudaStatus = cyclic_grids_transpose<32u, 8u>(f_prev, f_curr, f_tmp, x_size, y_size)))
				throw cuda_error_construct(cudaStatus, action);
			std::swap(f_prev, f_curr);
			std::swap(f_curr, f_tmp);
			grid_pointers_calculation(x_size + 1);
		}

		void y_to_x_transpose(std::string const &action) {
			cudaError_t cudaStatus;

			if (cudaSuccess != (cudaStatus = cyclic_grids_transpose<32u, 8u>(f_prev, f_curr, f_tmp, y_size, x_size)))
				throw cuda_error_construct(cudaStatus, action);
			std::swap(f_prev, f_curr);
			std::swap(f_curr, f_tmp);
			grid_pointers_calculation(y_size + 1);
		}

		void step() {
			
			forward_step();
			x_to_y_transpose("Forward step grid transpose");

			correction_step();
			y_to_x_transpose("Correction step grid transpose");

			forward_step_with_mixed_terms_correction();
			x_to_y_transpose("Forward step with mixed terms correction grid transpose");

			correction_step();
			y_to_x_transpose("Correction step afterm mixed terms correction grid transpose");

			std::swap(f_prev, f_curr);
		}

		~TwoDimensionalSolver() noexcept {
			cudaFree(global_device_memory);
		}

		void grid_pointers_calculation(size_t shift) {
			f_prev_grid = f_prev + shift;
			f_curr_grid = f_curr + shift;
			a = m + shift;
			b = a + x_size * y_size;
			c = b + x_size * y_size;
			d = c + x_size * y_size;
		}

		std::ostream &debug_stream;
		size_t const x_size, y_size, matrix_size;
		T *global_device_memory;
		T *f_prev, *f_curr, *f_tmp, *f_prev_grid, *f_curr_grid;
		T *dfc, *xx_dfc, *yy_dfc, *xy_dfc, *yx_dfc;
		T *m, *a, *b, *c, *d;
		T const rx, ry, rxy;
	};
} /* diffusion */ } /* iki */

#endif /* TwoDimensionalDeviceSolver_CUH */