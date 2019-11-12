#pragma once

namespace iki {
	template <typename T>
	struct ZFuncRungeKuttaSequence {
		T operator()(size_t x_cnt, T y) const { //F(x,y) = -x * y - T(1.)
			T x = x_cnt * dx;
			T k1 = -x * y - T(1);
			T k2 = -(x + T(0.5) * dx) * (y + k1 * T(0.5) * dx) - T(1);
			T k3 = -(x + T(0.5) * dx) * (y + k2 * T(0.5) * dx ) - T(1);
			T k4 = -(x + T(0.5) * dx) * (y + k2 * dx) - T(1);
			return dx / T(6) * (k1 + T(2) * k2 + T(2) * k3 + k4);
		}

		ZFuncRungeKuttaSequence(T dx): dx(dx) { }

	private:
		T dx;
	};
} /* iki */