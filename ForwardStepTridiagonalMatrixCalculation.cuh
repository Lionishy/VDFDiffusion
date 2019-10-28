#pragma once
#ifndef ForwardStepTridiagonalMatrixCalculation_CUH
#define ForwardStepTridiagonalMatrixCalculation_CUH

#include "DiagonalDiscretization.cuh"
#include "MixedTermDiscretization.cuh"
#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__device__ void forward_step_tridiagonal_matrix(T const *f, T const *xx_dfc, T const *yy_dfc, T const *xy_dfc, T const *yx_dfc, T *a, T *b, T *c, T *d, T rx, T ry, T rxy, size_t size, size_t x_stride, size_t y_stride) {
		a[0] = T(0);
		b[0] = (xx_dfc[-x_stride] + xx_dfc[0]) * rx / 2 + T(1);
		c[0] = -xx_dfc[0] * rx / 2;
		d[0] = 
			f[0] 
			+ rx / 2 * (diagonal_discretization(f[-x_stride], f[0], f[x_stride], xx_dfc[-x_stride], xx_dfc[0]) + f[-x_stride] * xx_dfc[-x_stride]) 
			+ ry * diagonal_discretization(f[-1], f[0], f[1], yy_dfc[-y_stride], yy_dfc[0]) 
			+ rxy * (
				mixed_term_discretization(f[-x_stride - 1], f[-1], f[x_stride - 1], f[-x_stride + 1], f[1], f[x_stride + 1], xy_dfc[-x_stride], xy_dfc[0]) 
				+ mixed_term_discretization(f[-x_stride - 1], f[-x_stride], f[1 - x_stride], f[x_stride - 1], f[x_stride], f[1 + x_stride], yx_dfc[-y_stride], yx_dfc[0])
			);

		for (size_t idx = 1; idx != size - 1; ++idx) {
			size_t stride_idx = idx * x_stride, stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * xx_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (xx_dfc[stride_prev] + xx_dfc[stride_idx]) + T(1);
			c[stride_idx] = -rx / 2 * xx_dfc[stride_idx];
			d[stride_idx] = 
				f[stride_idx] 
				+ rx / 2 * diagonal_discretization(f[stride_prev], f[stride_idx], f[stride_next], xx_dfc[stride_prev], xx_dfc[stride_idx]) 
				+ ry * diagonal_discretization(f[stride_idx - 1], f[stride_idx], f[stride_idx + 1], yy_dfc[idx - y_stride], yy_dfc[idx]) 
				+ rxy * (
					mixed_term_discretization(f[stride_prev - 1], f[stride_idx - 1], f[stride_next - 1], f[stride_prev + 1], f[stride_idx + 1], f[stride_next + 1], xy_dfc[stride_prev], xy_dfc[stride_idx])
					+ mixed_term_discretization(f[stride_prev - 1], f[stride_prev], f[+1 + stride_prev], f[stride_next - 1], f[stride_next], f[1 + stride_next], yx_dfc[idx - y_stride], yx_dfc[idx])
				);
		}

		{
			size_t stride_idx = x_stride * (size - 1), stride_prev = stride_idx - x_stride, stride_next = stride_idx + x_stride;
			a[stride_idx] = -rx / 2 * xx_dfc[stride_prev];
			b[stride_idx] = rx / 2 * (xx_dfc[stride_prev] + xx_dfc[stride_idx]) + T(1);
			c[stride_idx] = T(0);
			d[stride_idx] = 
				f[stride_idx] 
				+ rx / 2 * (diagonal_discretization(f[stride_prev], f[stride_idx], f[stride_next], xx_dfc[stride_prev], xx_dfc[stride_idx]) + f[stride_next] * xx_dfc[stride_idx]) 
				+ ry * diagonal_discretization(f[stride_idx - 1], f[stride_idx], f[stride_idx + 1],yy_dfc[size-1-y_stride],yy_dfc[size-1])
				+ rxy * (
					mixed_term_discretization(f[stride_prev - 1], f[stride_idx - 1], f[stride_next - 1], f[stride_prev + 1], f[stride_idx + 1], f[stride_next + 1], xy_dfc[stride_prev], xy_dfc[stride_idx])
					+ mixed_term_discretization(f[stride_prev - 1], f[stride_prev], f[1 + stride_prev], f[stride_next - 1], f[stride_next], f[+1 + stride_next], yx_dfc[size - 1 - y_stride], yx_dfc[size-1])
				);
		}
	}
} /* device */ } /* math */ } /* iki */

#endif /* ForwardStepTridiagonalMatrixCalculation_CUH */