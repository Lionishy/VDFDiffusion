#pragma once
#ifndef VDFmu_H
#define VDFmu_H

#include "Pow.h"
#include "PhysicalParameters.h"
#include "SimpleTable.h"

#include <cmath>

namespace iki { namespace whfi {
	template <typename T>
	struct VDFmu final {
	public:
		VDFmu(PhysicalParameters<T> params) : p(params) { }

		T operator()(T mu, T vparall) const {
			T coeff_c = std::exp(-mu), coeff_h = std::exp(-mu * p.TcTh_ratio);
			return
				p.nc * coeff_c * std::exp(-T(0.5) * math::pow<2>(vparall - p.bulk_to_term_c))
				+ p.nh * math::pow<3>(std::sqrt(p.TcTh_ratio)) * coeff_h *
				std::exp(-T(0.5) * math::pow<2>(vparall * std::sqrt(p.TcTh_ratio) - p.bulk_to_term_h));
		}

	private:
		PhysicalParameters<T> p;
	};

	template <typename T>
	struct VDFmuUniformGridTabulator final {
		VDFmuUniformGridTabulator(PhysicalParameters<T> params) : vdf(params) { }

		//sizes: 0 - vparall, 1 - vperp
		UniformSimpleTable<T, 2u, 1u> vperp_near(UniformSimpleTable<T, 2u, 1u> &table) {
			for (unsigned vparall_counter = 0u; vparall_counter != table.bounds.components[0]; ++vparall_counter) {
				for (unsigned vperp_counter = 0u; vperp_counter != table.bounds.components[1]; ++vperp_counter) {
					table.data[vperp_counter + vparall_counter * table.bounds.components[1]] =
						vdf(
							table.space.axes[1].begin + table.space.axes[1].step * vperp_counter
							, table.space.axes[0].begin + table.space.axes[0].step * vparall_counter
						);
				}
			}
			return table;
		}

		//sizes: 0 - vparall, 1 - vperp
		UniformSimpleTable<T, 2u, 1u> vparall_near(UniformSimpleTable<T, 2u, 1u> &table) {
			for (unsigned vperp_counter = 0u; vperp_counter != table.bounds.components[1]; ++vperp_counter) {
				for (unsigned vparall_counter = 0u; vparall_counter != table.bounds.components[0]; ++vparall_counter) {
					table.data[vparall_counter + vperp_counter * table.bounds.components[0]] =
						vdf(
							table.space.axes[1].begin + table.space.axes[1].step * vperp_counter
							, table.space.axes[0].begin + table.space.axes[0].step * vparall_counter
						);
				}
			}
			return table;
		}

	private:
		VDFmu<T> vdf;
	};
} /*whfi*/ } /*iki*/

#endif /*VDFmu_H*/