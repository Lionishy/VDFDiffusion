#pragma once
#ifndef StridedDiffusionStep_CUH
#define StridedDiffusionStep_CUH

#include <cuda_runtime.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T>
	__device__ void strided_diffusion_step(T *f, T *dfc, T *a, T *b, T *c, T *d, T r, size_t size, size_t stride) {
		strided_calculate_tridiagonal_matrix(f, dfc, a, b, c, d, r, size, stride);
		iki::math::device::thomson_sweep(a, b, c, d, f, size - 1, stride);
	}
} /* device */ } /* diffusion */ } /* iki */

#endif /* DiffusionStep_CUH */