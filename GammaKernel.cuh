#pragma once
#ifndef GammaKernel_CUH
#define GammaKernel_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device {
	template <typename T>
	__global__ void gamma_kernel(T const *zero_moment, T const *first_moment, T const *k_betta, T const *dispersion_derive, T vparall_step, unsigned size, T *gamma) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		if (0 == shift || size - 1 == shift) {
			gamma[shift] = 0; return;
		}
		T first_moment_derive = T(0.5) * (first_moment[shift + 1] - first_moment[shift - 1]) / vparall_step;
		//sqrt(pi/2) = 1.25331414
		gamma[shift] = -T(1.25331414) / k_betta[shift] * (k_betta[shift] * first_moment_derive - zero_moment[shift]) / dispersion_derive[shift];
	}
} /* device */ } /* whfi */ } /* iki */

#endif /* GammaKernel_CUH */