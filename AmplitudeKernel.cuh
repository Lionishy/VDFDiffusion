#pragma once
#ifndef AmplitudeKernle_CUH
#define AmplitudeKernle_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki {	namespace whfi { namespace device {
	template <typename T>
	__global__ void amplitude_update_kernell(T const *gamma, T *amplitude_spectrum, T dt, unsigned size) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		if (gamma[shift] < T(0.))
			amplitude_spectrum[shift] = T(0.);
		else
			amplitude_spectrum[shift] += T(2.0) * dt * gamma[shift] * amplitude_spectrum[shift];
	}
} /*device*/ } /*whfi*/ } /*iki*/

#endif /*AmplitudeKernle_CUH*/