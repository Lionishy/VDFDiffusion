#pragma once
#ifndef ForwardStepTridiagonalMatrixCorrection_CUH
#define ForwardStepTridiagonalMatrixCorrection_CUH

#include "MixedTermDiscretization.cuh"
#include <cuda_runtime.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T>
	__device__ void forward_step_tridiagonal_matrix_correction(T const *f_prev, T const *f_curr, T const *xy_dfc, T const *yx_dfc, T *d, T rxy, size_t size, size_t x_stride, size_t y_stride) {
		for (size_t idx = 0; idx != size; ++idx) {
			size_t stride_idx = idx * x_stride, stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			d[stride_idx] =
				rxy / 2 * (
					mixed_term_discretization(f_curr[stride_prev - 1], f_curr[stride_idx - 1], f_curr[stride_next - 1], f_curr[stride_prev + 1], f_curr[stride_idx + 1], f_curr[stride_next + 1], xy_dfc[stride_prev], xy_dfc[stride_idx])
					+ mixed_term_discretization(f_curr[-1 + stride_prev], f_curr[stride_prev], f_curr[+1 + stride_prev], f_curr[-1 + stride_next], f_curr[stride_next], f_curr[+1 + stride_next], yx_dfc[idx - y_stride], yx_dfc[idx])

					- mixed_term_discretization(f_prev[stride_prev - 1], f_prev[stride_idx - 1], f_prev[stride_next - 1], f_prev[stride_prev + 1], f_prev[stride_idx + 1], f_prev[stride_next + 1], xy_dfc[stride_prev], xy_dfc[stride_idx])
					- mixed_term_discretization(f_prev[-1 + stride_prev], f_prev[stride_prev], f_prev[+1 + stride_prev], f_prev[-1 + stride_next], f_prev[stride_next], f_prev[+1 + stride_next], yx_dfc[idx - y_stride], yx_dfc[idx])
				);
		}
	}
} /* device */ } /* math */ } /* iki */

#endif /* ForwardStepTridiagonalMatrixCorrection_CUH */