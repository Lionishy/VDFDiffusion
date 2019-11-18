#pragma once
#ifndef ZeroMomentKernel_CUH
#define ZeroMomentKernel_CUH

#include "ZeroMoment.cuh"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace iki { namespace math { 
	template <typename T>
	__global__ void zero_moment_kernel(T const *f, T start, T dx, unsigned x_size, unsigned y_size, T *zero_moment) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		T s, rem;
		iki::math::device::zero_moment(f + shift + y_size, start, dx, x_size - 2, y_size, &s, &rem);
		s -= T(0.5) * (*(f + shift) + *(f + shift + y_size * (x_size - 1)));
		*(zero_moment + shift) = s * dx;
	}
} /* math */ } /* iki */

#endif /* ZeroMomentKernel_CUH */