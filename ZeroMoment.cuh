#pragma once
#ifndef ZeroMoment_CUH
#define ZeroMoment_CUH

#include <cuda_runtime.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	void zero_moment(T const *f, unsigned size, unsigned stride, T *s, T *r) {

		T sum = T(0), rem = T(0);
		for (unsigned idx = 0, end = size * stride; idx != end; idx += stride)
			sum += f[idx];
		*s = sum;
		*r = rem;
	}
} /* device */ } /* math */ } /* iki */


#endif /* ZeroMoment_CUH */