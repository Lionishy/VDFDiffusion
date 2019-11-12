#pragma once
#ifndef FirstMoment_CUH
#define FirstMoment_CUH

#include <cuda_runtime.h>

namespace iki {	namespace math { namespace device {
	template <typename T>
	__device__ void first_moment(T const *f, T start, T dx, unsigned size, unsigned stride, T *s, T *r) {
		T sum = T(0), rem = T(0);
		for (unsigned fidx = 0, fend = size * stride, arg_idx = 0; fidx != fend; fidx += stride, ++arg_idx) {
			T y, t;
			y = (start + dx * arg_idx) * f[fidx] - rem;
			t = sum + y;
			rem = (t - sum) - y;
			sum = t;
		}
		*s = sum;
		*r = rem;
	}
} /* device */ } /* math */ } /* iki */


#endif /* FirstMoment_CUH */