#pragma once
#ifndef AnalyticalMoments_H
#define AnalyticalMoments_H

#include "PhysicalParameters.h"
#include "Pow.h"

#include <vector>
#include <cmath>

namespace iki { namespace whfi { 
	template <typename T>
	class AnalyticalMoments final {
	public:
		AnalyticalMoments(PhysicalParameters<T> p) : p(p) { }

		std::vector<T> g(T vparall_begin, T vparall_step, unsigned size) const {
			auto g_vparall = [this] (T vparall) {
				return p.nc * std::exp(-math::pow<2>(vparall - p.bulk_to_term_c) / T(2)) + p.nh * std::sqrt(p.TcTh_ratio) * exp(-math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2));
			};
			std::vector<T> table(size);
			for (unsigned count = 0u; count < size; ++count)
				table[count] = g_vparall(vparall_begin + vparall_step * count);
			return table;
		}

		std::vector<T> G(T vparall_begin, T vparall_step, unsigned size) const {
			auto G_vparall = [this] (T vparall) {
				return p.nc * std::exp(-pow<2>(vparall - p.bulk_to_term_c) / T(2)) + p.nh * std::sqrt(T(1) / p.TcTh_ratio) * exp(-math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h) / T(2));
			};
			std::vector<T> table(size);
			for (unsigned count = 0u; count < size; ++count)
				table[count] = G_vparall(vparall_begin + vparall_step * count);
			return table;
		}

	private:
		iki::whfi::PhysicalParameters<T> p;
	};
} /* whfi */ } /* iki */

#endif /* AnalyticalMoments_H */