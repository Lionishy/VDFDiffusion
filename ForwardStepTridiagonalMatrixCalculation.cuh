#pragma once
#ifndef ForwardStepTridiagonalMatrixCalculation_CUH
#define ForwardStepTridiagonalMatrixCalculation_CUH

#include "DiagonalDiscretization.cuh"
#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__device__ void forward_step_tridiagonal_matrix(T const *f, T const *x_dfc, T const *y_dfc, T *a, T *b, T *c, T *d, T rx, T ry, size_t size, size_t x_stride, size_t y_stride) {
		a[0] = T(0);
		b[0] = (x_dfc[-x_stride] + x_dfc[0]) * rx / 2 + T(1);
		c[0] = -x_dfc[0] * rx / 2;
		d[0] = f[0] + rx / 2 * (diagonal_discretization(f[-x_stride], f[0], f[x_stride], x_dfc[-x_stride], x_dfc[0]) + f[-x_stride] * x_dfc[-x_stride]) + ry * diagonal_discretization(f[-1],f[0],f[1],y_dfc[-y_stride],y_dfc[0]);

		for (size_t idx = 1; idx != size - 1; ++idx) {
			size_t stride_idx = idx * x_stride, stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * x_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (x_dfc[stride_prev] + x_dfc[stride_idx]) + T(1);
			c[stride_idx] = -rx / 2 * x_dfc[stride_idx];
			d[stride_idx] = f[stride_idx] + rx / 2 * diagonal_discretization(f[stride_prev], f[stride_idx], f[stride_next], x_dfc[stride_prev], x_dfc[stride_idx]) + ry * diagonal_discretization(f[stride_idx-1],f[stride_idx],f[stride_idx+1],y_dfc[idx-y_stride],y_dfc[idx]);
		}

		{
			size_t stride_idx = x_stride * (size - 1), stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * x_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (x_dfc[stride_prev] + x_dfc[stride_idx]) + T(1);
			c[stride_idx] = T(0);
			d[stride_idx] = f[stride_idx] + rx / 2 * (diagonal_discretization(f[stride_prev], f[stride_idx], f[stride_next], x_dfc[stride_prev], x_dfc[stride_idx]) + f[stride_next] * x_dfc[stride_idx]) + ry * diagonal_discretization(f[stride_idx - 1], f[stride_idx], f[stride_idx + 1],y_dfc[size-1-y_stride],y_dfc[size-1]);
		}
	}
} /* device */ } /* math */ } /* iki */

#endif /* ForwardStepTridiagonalMatrixCalculation_CUH */