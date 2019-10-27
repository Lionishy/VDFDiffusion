#pragma once
#ifndef TwoDimensionalDeviceSolver_CUH
#define TWoDimensionalDeviceSolver_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace iki { namespace diffusion {
	template <typename T>
	struct TwoDimensionalSolver final {
		TwoDimensionalSolver(size_t x_size, size_t y_size): x_size(x_size), y_size(y_size), global_device_memory(NULL) {
			size_t bytes = 3 * x_size * y_size * sizeof(T) + 4 * x_size * y_size * sizeof(T) + 4 * x_size * y_size * sizeof(T);
			cudaError_t cudaStatus;
			if (cudaSuccess != (cudaStatus = cudaMalloc(&global_device_memory, bytes))) {
				std::stringstream error_stream;
				error_stream << "Can't allocate global device memory about " << ((bytes - 1) / 1024 + 1) "Kb in total.\n" << "Device code: " << cudaStatus << "\n" << "Device string: " << cudaGetErrorString(cudaStatus);
				throw std::runtime_error(error_stream.str());
			}
		}

		~TwoDimensionalSolver() noexcept {
			cudaFree(global_device_memory);
		}

		size_t x_size, y_size;
		T *global_device_memory;
		T *f_prev, *f_curr, *f_tmp;
		T *xx_dfc, *yy_dfc, *xy_dfc, *yx_dfc;
		T *a, *b, *c, *d;
	};
} /* diffusion */ } /* iki */

#endif /* TWoDimensionalDeviceSolver_CUH */