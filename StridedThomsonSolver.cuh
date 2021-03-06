#pragma once
#ifndef StridedThomsonSolver_CUH
#define StridedThomsonSolver_CUH

#include <cuda_runtime.h>

namespace iki {	namespace math { namespace device {
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

	template <>
	__device__ void strided_thomson_sweep<float>(float *a, float *b, float *c, float *d, float *x, size_t size, size_t stride) {
		for (size_t idx = 1; idx != size; ++idx) {
			size_t stride_idx = idx * stride;
			float w = -a[stride_idx] / b[stride_idx - stride];
			b[stride_idx] = __fmaf_rn(w, c[stride_idx - stride], b[stride_idx]);
			d[stride_idx] = __fmaf_rn(w, d[stride_idx - stride], d[stride_idx]);
		}
		x[stride * (size - 1)] = d[stride * (size - 1)] / b[stride * (size - 1)];

		for (size_t idx = size - 2; idx != 0; --idx) {
			size_t stride_idx = idx * stride;
			x[stride_idx] = __fmaf_rn(-c[stride_idx], x[stride_idx + stride], d[stride_idx]) / b[stride_idx];
		}
		x[0] = __fmaf_rn(-c[0], x[stride], d[0]) / b[0];
	}
}/* device */ } /* math */ } /* iki */

#endif