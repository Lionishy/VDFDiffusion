#pragma once
#ifndef DiffusionCoefficientMultiplicationKernel_CUH
#define DiffusionCoefficientMultiplicationKernel_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace whfi { namespace device { 
	template <typename T>
	__global__ void diffusion_coefficient_multiplication_kernell(T *dfc, T const *dfc_pivot, T const *amplitude_spectrum, size_t vperp_size, size_t vparall_size) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		for (size_t vpar_idx = 0; vpar_idx != vparall_size; ++vpar_idx) {
			dfc[vpar_idx * vperp_size + shift] = dfc_pivot[vpar_idx * vperp_size + shift] * amplitude_spectrum[vpar_idx];
		}
	}
} /*device*/ } /*whfi*/ } /*iki*/

#endif /*DiffusionCoefficientMultiplicationKernel_CUH*/