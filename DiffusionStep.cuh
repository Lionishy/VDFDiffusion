#pragma once
#ifndef DiffusionStep_CUH
#define DiffusionStep_CUH

#include <cuda_runtime.h>

namespace iki { namespace diffusion { namespace device {
	template <typename T>
	__device__ void diffusion_step(T *f, T *dfc, T *a, T *b, T *c, T *d, size_t size, T r) {
		calculate_tridiagonal_matrix(f, dfc, a, b, c, d, size, r);
		iki::math::device::thomson_sweep(a, b, c, d, f, size - 1);
	}
} /* device */ } /* diffusion */ } /* iki */

#endif /* DiffusionStep_CUH */