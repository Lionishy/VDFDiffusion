#ifndef ThomsonSolverKernel_CUH
#define ThomsonSolverKernel_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace iki { namespace math { namespace device { 
	template <typename T>
	__global__ void thomson_multisolver_kernell(T *a, T *b, T *c, T *d, T *x, size_t size, size_t span) {
		size_t shift = blockIdx.x * blockDim.x + threadIdx.x;
		strided_thomson_sweep(a + shift, b + shift, c + shift, d + shift, x + shift, size, span);
	}
} /* device */ } /* math */ } /* iki */

#endif /* ThomsonSolverKernel_CUH */