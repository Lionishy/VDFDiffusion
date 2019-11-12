#pragma once
#ifndef Pow_H
#define Pow_H

namespace iki { namespace math {
	template <unsigned P, typename T>
	inline
		T pow(T x) {
		T accum = T(1);
		for (unsigned count = 0; count != P; ++count)
			accum *= x;
		return accum;
	}
} /* math */ } /* iki */

#endif /* Pow_H */
