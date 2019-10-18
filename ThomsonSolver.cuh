#pragma once
#ifndef ThomsonSolver_CUH
#define ThomsonSolver_CUH

#include <cuda_runtime.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__device__ void thomson_sweep(T *a, T *b, T *c, T *d, T *x, size_t size) {
		for (size_t idx = 1; idx != size; ++idx) {
			T w = a[idx] / b[idx-1];
			b[idx] = b[idx] - w * c[idx - 1];
			d[idx] = d[idx] - w * d[idx - 1];
		}
		x[size - 1] = d[size - 1] / b[size - 1];

		for (size_t idx = size - 2; idx != 0; --idx)
			x[idx] = (d[idx] - c[idx] * x[idx + 1]) / b[idx];
		x[0] = (d[0] - c[0] * x[1]) / b[0];
	}

	template <>
	__device__ void thomson_sweep<float>(float *a, float *b, float *c, float *d, float *x, size_t size) {
		for (size_t idx = 1; idx != size; ++idx) {
			float w = -a[idx] / b[idx - 1];
			b[idx] = __fmaf_rn(w,c[idx-1],b[idx]);
			d[idx] = __fmaf_rn(w,d[idx - 1],d[idx]);
		}
		x[size - 1] = d[size - 1] / b[size - 1];

		for (size_t idx = size - 2; idx != 0; --idx)
			x[idx] = __fmaf_rn(-c[idx],x[idx + 1],d[idx]) / b[idx];
		x[0] = __fmaf_rn(-c[0],x[1],d[0]) / b[0];
	}
}/* device */ } /* math */ } /* iki */

#endif