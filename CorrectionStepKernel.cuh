#pragma once
#ifndef CorrectionStepKernel_CUH
#define CorrectionStepKernel_CUH

#include "CorrectionStepTridiagonalMatrixCalculation.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T>
	__global__ void correction_step_multisolver_kernel(T const *f_prev, T const *f_curr, T const *x_dfc, T *a, T *b, T *c, T *d, T rx, size_t size, size_t x_stride) {
		size_t shift = blockIdx.x * blockDim.x + threadIdx.x;
		correction_step_tridiagonal_matrix(f_prev + shift, f_curr + shift, x_dfc + shift, a + shift, b + shift, c + shift, d + shift, rx, size, x_stride);
	}
} /* device */ } /* diffusion */ } /* iki */

#endif /* CorrectionStepKernel_CUH */