#pragma once
#ifndef ForwardStepCorrectionKernell_CUH
#define ForwardStepCorrectionKernell_CUH


#include "ForwardStepTridiagonalMatrixCorrection.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T>
	__global__ void forward_step_correction_multisolver_kernel(T const *f_prev, T const *f_curr, T const *xy_dfc, T const *yx_dfc, T *d, T rxy, size_t size, size_t x_stride, size_t y_stride) {
		size_t shift = blockIdx.x * blockDim.x + threadIdx.x;
		forward_step_tridiagonal_matrix_correction(f_prev + shift, f_curr + shift, xy_dfc + shift, yx_dfc + shift * y_stride, d + shift, rxy, size, x_stride, y_stride);
	}

} /* device */ } /* diffusion */ } /* iki */

#endif