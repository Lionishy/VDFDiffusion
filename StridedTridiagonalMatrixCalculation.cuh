#pragma once
#ifndef StridedTridiagonalMatrixCalculation_CUH
#define StridedTridiagonalMatrixCalculation_CUH

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device { 
	template <typename T>
	__device__ void calculate_tridiagonal_matrix(T const *f, T const *dfc, T *a, T *b, T *c, T *d, T r, size_t size, size_t stride) {
		a[0] = T(0);
		b[0] = dfc[0] * r / 2 + T(1);
		c[0] = -dfc[0] * r / 2;
		d[0] = f[0] + r / 2 * dfc[0] * (f[stride] -f[0]);

		for (size_t idx = 1; idx != size - 2; ++idx) {
			size_t stride_idx = idx * stride, stride_prev = stride_idx - stride, stride_next = stride_idx + stride;
			a[stride_idx] = -r / 2 * dfc[stride_prev];
			b[stride_idx] = r / 2 * (dfc[stride_prev] + dfc[stride_idx]) + T(1);
			c[stride_idx] = -r / 2 * dfc[stride_idx];
			d[stride_idx] = f[stride_idx] + r / 2 * (f[stride_prev] * dfc[stride_prev] - f[stride_idx] * (dfc[stride_prev] + dfc[stride_idx]) + f[stride_next] * dfc[stride_idx]);
		}

		{
			T stride_idx = stride * (size - 2), stride_prev = stride_idx - stride, stride_next = stride_idx + stride;
			a[stride_idx] = -r / 2 * dfc[stride_prev];
			b[stride_idx] = r / 2 * (dfc[stride_prev] + dfc[stride_idx]) + T(1);
			c[stride_idx] = T(0);
			d[stride_idx = f[stride_idx] + r / 2 * (f[stride_prev] * dfc[stride_prev] - f[stride_idx] * (dfc[stride_prev] + dfc[stride_idx]) + 2 * f[stride_next] * dfc[stride_idx]);
		}
	}
} /* device */ } /* diffusion */ } /* iki */

#endif /* StridedTridiagonalMatrixCalculation_CUH */