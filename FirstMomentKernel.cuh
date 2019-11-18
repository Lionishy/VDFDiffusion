#pragma once
#ifndef FirstMomentKernel_CUH
#define FirstMomentKernel_CUH

#include "FirstMoment.cuh"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__global__ void first_moment_kernel(T const *f, T start, T dx, unsigned x_size, unsigned y_size, T *first_moment) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		T s, rem;
		iki::math::device::first_moment(f + shift + y_size, start, dx, x_size - 2, y_size, &s, &rem);
		s -= T(0.5) * (*(f + shift) * start + *(f + shift + y_size * (x_size - 1)) * (start + dx * (x_size - 1)));
		*(first_moment + shift) = s * dx;
	}
} /* device */ } /* math */ } /* iki */

#endif /* FirstMomentKernel_CUH */