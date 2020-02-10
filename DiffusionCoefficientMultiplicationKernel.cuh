#pragma once
#ifndef DiffusionCoefficientMultiplicationKernel_CUH
#define DiffusionCoefficientMultiplicationKernel_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device { 
	template <typename T>
	__global__ void diffusion_coefficient_multiplication_kernell(T *dfc, T const *dfc_pivot, T const *amplitude_spectrum, size_t vperp_size, size_t vparall_size) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		T const coeff = amplitude_spectrum[shift];
		for (size_t vperp_idx = 0; vperp_idx != vperp_size; ++vperp_idx) {
			dfc[vperp_idx * vparall_size + shift] = dfc_pivot[vperp_idx * vparall_size + shift] * coeff;
		}
	}
} /*device*/ } /*whfi*/ } /*iki*/

#endif /*DiffusionCoefficientMultiplicationKernel_CUH*/