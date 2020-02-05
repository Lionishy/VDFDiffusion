#pragma once
#ifndef DispersionRelationOmegaDerive_H
#define DispersionRelationOmegaDerive_H

#include "PhysicalParameters.h"
#include "ZFunc.h"

namespace iki { namespace whfi { 
	template <typename T>
	struct DispersionRelationOmegaDerivative {
		T operator()(T omega, T k) {
			T arg_c = (omega - T(1.)) / (k * params.betta_root_c) - params.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k * params.betta_root_h) - params.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return params.nc / (k * params.betta_root_c) * (-Zc + (omega / (k * params.betta_root_c) - params.bulk_to_term_c) * (Zc * arg_c + T(1.)))
				+ params.nh / (k * params.betta_root_h) * (-Zh + (omega / (k * params.betta_root_h) - params.bulk_to_term_h) * (Zh * arg_h + T(1.)));
		}

		DispersionRelationOmegaDerivative(ZFunc Z, PhysicalParameters<T> params) : Z(Z), params(params) { }

	private:
		ZFunc Z;
		PhysicalParameters<T> params;
	};
} /*whfi*/ } /*iki*/

#endif /*DispersionRelationOmegaDerive_H*/