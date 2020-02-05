#pragma once
#ifndef DispersionRelationKDerive_H
#define DispersionRelationKDerive_H

#include "PhysicalParameters.h"
#include "ZFunc.h"

namespace iki { namespace whfi { 
	template <typename T>
	struct DispersionRelationKDerivative {
		T operator()(T omega, T k_betta_c) {
			T k_betta_h = k_betta_c / params.TcTh_ratio_root;
			T k = k_betta_c / params.betta_root_c;
			T arg_c = (omega - T(1.)) / (k_betta_c) - params.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k_betta_h) - params.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return T(2.) * k
				+ params.nc * (omega / (k * k_betta_c) + params.bulk_to_term_c) * Zc
				- params.nc * (omega / (k_betta_c) - params.bulk_to_term_c) * (Zc * arg_c + T(1.)) * (omega - T(1.)) / (k * k_betta_c)
				+ params.nh * (omega / (k * k_betta_h) + params.bulk_to_term_h) * Zh
				- params.nh * (omega / (k_betta_h) - params.bulk_to_term_h) * (Zh * arg_h + T(1.)) * (omega - T(1.)) / (k * k_betta_h);
		}

		DispersionRelationKDerivative(ZFunc Z, PhysicalParameters<T> params) : Z(Z), params(params) { }

	private:
		ZFunc Z;
		PhysicalParameters<T> params;
	};
} /*whfi*/ } /*iki*/

#endif /*DispersionRelationKDerive_H*/