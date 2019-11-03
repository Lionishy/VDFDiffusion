#pragma once
#ifndef ForwardStepKernel_CUH
#define ForwardStepKernel_CUH

#include "ForwardStepTridiagonalMatrixCalculation.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__global__ void forward_step_multisolver_kernel(T const *f_curr, T const *x_dfc, T const *y_dfc, T const *xy_dfc, T const *yx_dfc, T *a, T *b, T *c, T *d, T rx, T ry, T rxy, size_t size, size_t x_stride, size_t y_stride) {
		size_t shift = blockIdx.x * blockDim.x + threadIdx.x;
		forward_step_tridiagonal_matrix(f_curr + shift, x_dfc + shift, y_dfc + shift * y_stride, xy_dfc + shift, yx_dfc + shift * y_stride,  a + shift, b + shift, c + shift, d + shift, rx, ry, rxy, size, x_stride, y_stride);
	}

} /* device */ } /* diffusion */ } /* iki */

#endif /* ForwardStepKernel_CUH */