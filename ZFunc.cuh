#pragma once
#ifndef ZFunc_CUH
#define ZFunc_CUH

#include <cuda_runtime.h>

namespace iki {
	template <typename T>
	struct ZFunc {
		__device__ T operator()(T arg) const {
			T farg = fabs(arg);
			auto idx = size_t(farg / step);
			if ((idx + 1u) < size) {
				return (arg > T(0.) ? T(1) : T(-1)) *((table[idx + 1u] - table[idx]) / step * (farg - step * idx) + table[idx]);
			}
			else { //asymptotic
				T over = T(1.) / arg, square = over * over;
				return -over * (T(1) + square + T(3) * square * square);
			}
		}

		__device__ ZFunc(T step, unsigned size, T *table): step(step), size(size), table(table) { }

		T step;
		unsigned size;
		T *table;
	};
} /* iki */

#endif /* ZFunc_CUH */