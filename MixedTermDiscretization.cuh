#pragma once
#ifndef MixedTermDiscretization_CUH
#define MixedTermDiscretization_CUH

#include <cuda_runtime.h>

namespace iki {	namespace diffusion { namespace device {
	template <typename T> inline
		T __device__ mixed_term_discretization(T Fim1jm1, T Fijm1, T Fip1jm1, T Fim1jp1, T Fijp1, T Fip1jp1, T xy_dfc_left, T xy_dfc_right) {
		return T(0.25) * (xy_dfc_right * (Fijp1 + Fip1jp1 - Fijm1 - Fip1jm1) - xy_dfc_left * (Fijp1 + Fim1jp1 - Fijm1 - Fim1jm1));
	}

	template <> inline
		float __device__ mixed_term_discretization<float>(float Fim1jm1, float Fijm1, float Fip1jm1, float Fim1jp1, float Fijp1, float Fip1jp1, float xy_dfc_left, float xy_dfc_right) {
		return 0.25f * (__fmaf_rn(xy_dfc_right, Fijp1 + Fip1jp1, -__fmaf_rn(xy_dfc_right, Fijm1, xy_dfc_right * Fip1jm1)) - __fmaf_rn(xy_dfc_left, Fijp1 + Fim1jp1, -__fmaf_rn(xy_dfc_right, Fijm1, xy_dfc_right * Fim1jm1)));
	}
} /* device */	} /* math */ } /* iki */

#endif /* MixedTermDiscretization_CUH */