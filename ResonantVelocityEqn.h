#pragma once
#ifndef ResonantVelocityEqn_H
#define ResonantVelocityEqn_H

#include "PhysicalParameters.h"
#include "ZFunc.h"

namespace iki { namespace whfi {
	/**
	 * The class representing a dispersion equation for the whistler wave 
	 * as a function of the omega only as we recalcuate the wave number 
	 * from the resonant condition vR = (omega - 1)/(k * sqrt_betta_core)
	 */
	template <typename T>
	struct ResonantVelocityEqn {
		T operator()(T omega) const {
			T Zc = Z(v_res - params.bulk_to_term_c)
				, Zh = Z(v_res * sqrt(params.TcTh_ratio) - params.bulk_to_term_h);
			return T(1. / 1836.)
				+ (omega - T(1)) * (omega - T(1)) / (v_res * params.betta_root_c) / (v_res * params.betta_root_c)
				- params.nc * ((omega * v_res) / (omega - T(1)) - params.bulk_to_term_c) * Zc
				- params.nh * ((omega * v_res * sqrt(params.TcTh_ratio)) / (omega - T(1)) - params.bulk_to_term_h) * Zh;
		}

		ResonantVelocityEqn(T v_res, iki::whfi::PhysicalParameters<T> params, iki::ZFunc<T> Z) : v_res(v_res), params(params), Z(Z) { }

		T v_res;
		PhysicalParameters<T> params;
		ZFunc<T> Z;
	};
} /*whif*/ } /*iki*/

#endif /*ResonantVelocityEqn_H*/