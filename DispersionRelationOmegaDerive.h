#pragma once
#ifndef DispersionRelationOmegaDerive_H
#define DispersionRelationOmegaDerive_H

#include "PhysicalParameters.h"
#include "ZFunc.h"

namespace iki { namespace whfi { 
	template <typename T>
	struct DispersionRelationOmegaDerivative {
		T operator()(T omega, T k_betta_c) {
			T k_betta_h = k_betta_c / params.TcTh_ratio_root;
			T arg_c = (omega - T(1.)) / (k_betta_c) - params.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k_betta_h) - params.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return params.nc / (k_betta_c) * (-Zc + (omega / (k_betta_c) - params.bulk_to_term_c) * (Zc * arg_c + T(1.)))
				+ params.nh / (k_betta_h) * (-Zh + (omega / (k_betta_h) - params.bulk_to_term_h) * (Zh * arg_h + T(1.)));
		}

		DispersionRelationOmegaDerivative(ZFunc<T> Z, PhysicalParameters<T> params) : Z(Z), params(params) { }

	private:
		ZFunc<T> Z;
		PhysicalParameters<T> params;
	};
} /*whfi*/ } /*iki*/

#endif /*DispersionRelationOmegaDerive_H*/