#pragma once
#ifndef CorrectionStepTridiagonalMatrixCalculation_CUH
#define CorrectionStepTridiagonalMatrixCalculation_CUH

#include "DiagonalDiscretization.cuh"
#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__device__ void correction_step_tridiagonal_matrix(T const *f_prev, T const *f_curr, T const *x_dfc, T *a, T *b, T *c, T *d, T rx, size_t size, size_t x_stride) {
		a[0] = T(0);
		b[0] = (x_dfc[-x_stride] + x_dfc[0]) * rx / 2 + T(1);
		c[0] = -x_dfc[0] * rx / 2;
		d[0] = f_curr[0] - rx / 2 * diagonal_discretization(f_prev[-x_stride], f_prev[0], f_prev[x_stride], x_dfc[-x_stride], x_dfc[0]);

		for (size_t idx = 1; idx != size - 1; ++idx) {
			size_t stride_idx = idx * x_stride, stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * x_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (x_dfc[stride_prev] + x_dfc[stride_idx]) + T(1);
			c[stride_idx] = -rx / 2 * x_dfc[stride_idx];
			d[stride_idx] = f_curr[stride_idx] - rx / 2 * diagonal_discretization(f_prev[stride_prev], f_prev[stride_idx], f_prev[stride_next], x_dfc[stride_prev], x_dfc[stride_idx]);
		}

		{
			size_t stride_idx = x_stride * (size - 1), stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * x_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (x_dfc[stride_prev] + x_dfc[stride_idx]) + T(1);
			c[stride_idx] = T(0);
			d[stride_idx] = f[stride_idx] - rx / 2 * (diagonal_discretization(f_prev[stride_prev], f_prev[stride_idx], f_prev[stride_next], x_dfc[stride_prev], x_dfc[stride_idx]) - T(0.5)*(f_prev[stride_next] + f_curr[stride_next]) * x_dfc[stride_idx]);
		}

	}
} /* device */ } /* diffusion */ } /* iki */

#endif /* CorrectionStepTridiagonalMatrixCalculation_CUH */