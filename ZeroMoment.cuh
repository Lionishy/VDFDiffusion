#pragma once
#ifndef ZeroMoment_CUH
#define ZeroMoment_CUH

#include <cuda_runtime.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__device__ void zero_moment(T const *f, T start, T dx, unsigned size, unsigned stride, T *s, T *r) {

		T sum = T(0), rem = T(0), y, t;
		for (unsigned idx = 0, end = size * stride; idx != end; idx += stride) {
			y = f[idx] - rem;
			t = sum + y;
			rem = (t - sum) - y;
			sum = t;
		}
		*s = sum;
		*r = rem;
	}
} /* device */ } /* math */ } /* iki */


#endif /* ZeroMoment_CUH */