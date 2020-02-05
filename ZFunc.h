#pragma once
#ifndef ZFunc_H
#define ZFunc_H

namespace iki {
	template <typename T>
	struct ZFunc {
		T operator()(T arg) const {
			T farg = arg < T(0.) ? -arg : arg;
			auto idx = size_t(farg / step);
			if ((idx + 1u) < size) {
				return (arg > T(0.) ? T(1) : T(-1)) *((table[idx + 1u] - table[idx]) / step * (farg - step * idx) + table[idx]);
			}
			else { //asymptotic
				T over = T(1.) / arg, square = over * over;
				return -over * (T(1) + square + T(3) * square * square);
			}
		}

		ZFunc(T step, unsigned size, T const *table) : step(step), size(size), table(table) { }

		T step;
		unsigned size;
		T const *table;
	};
} /* iki */

#endif /* ZFunc_H */