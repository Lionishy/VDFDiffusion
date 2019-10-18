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

	template <typename T>
	__device__ void strided_thomson_sweep(T *a, T *b, T *c, T *d, T *x, size_t size, size_t stride) {
		for (size_t idx = 1; idx != size; ++idx) {
			size_t stride_idx = idx * stride;
			T w = a[stride_idx] / b[stride_idx - stride];
			b[stride_idx] = b[stride_idx] - w * c[stride_idx - stride];
			d[stride_idx] = d[stride_idx] - w * d[stride_idx - stride];
		}
		x[stride * (size - 1)] = d[stride * (size - 1)] / b[stride * (size - 1)];

		for (size_t idx = size - 2; idx != 0; --idx) {
			size_t stride_idx = idx * stride;
			x[stride_idx] = (d[stride_idx] - c[stride_idx] * x[stride_idx + stride]) / b[stride_idx];
		}
		x[0] = (d[0] - c[0] * x[stride]) / b[0];
	}
}/* device */ } /* math */ } /* iki */

#endif