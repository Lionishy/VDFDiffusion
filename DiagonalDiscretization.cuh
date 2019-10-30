#pragma once
#ifndef DiagonalDiscretization_CUH
#define DiagonalDiscretization_CUH

#include <cuda_runtime.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T> inline
		T __device__ diagonal_discretization(T fim1, T fi, T fip1, T dfc_left, T dfc_right) {
		return dfc_left * (fim1 - fi) + dfc_right * (fip1 - fi);
	}

	template <> inline
		float __device__ diagonal_discretization<float>(float fim1, float fi, float fip1, float dfc_left, float dfc_right) {
		return __fmaf_rn(dfc_left, fim1, -dfc_left * fi) + __fmaf_rn(dfc_right, fip1, -dfc_right * fi);
	}
} /* device */ } /* math */ } /* iki */

#endif /* DiagonalDiscretization_CUH */