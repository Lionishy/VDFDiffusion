#pragma once
#ifndef ThomsonSolver_CUH
#define ThomsonSolver_CUH

#include <cuda_runtime.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__device__ void thomson_sweep(T const *a, T const *b, T *c, T *d, T *x, size_t size) {
		for (size_t idx = 1; idx != size; ++idx) {
			T w = a[idx] / b[idx];
			b[idx] = b[idx] - w * c[idx - 1];
			d[idx] = d[idx] - w * d[idx - 1];
		}
		x[size - 1] = d[size - 1] / b[size - 1];

		for (size_t idx = size - 2; idx != 0; --idx)
			x[idx] = (d[idx] - c[idx] * x[idx + 1]) / b[idx];
		x[0] = (d[0] - c[0] * x[1]) / b[0];
	}
}/* device */ } /* math */ } /* iki */

#endif