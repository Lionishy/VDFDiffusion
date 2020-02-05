#pragma once
#ifndef InitialConditionsClass_CUH
#define InitialConditionsClass_CUH

#include "ZFunc.h"
#include "SimpleTable.h"
#include "PhysicalParameters.h"

#include <vector>

namespace iki { namespace whfi {
	template <typename T>
	struct InitialConditions {
		std::vector<T> omega, k_betta, dispersion_derivative, Dpr, Dpl, Dmx;
	};

	template <typename T>
	struct InitialConditionsCalculation {
		InitialConditions<T> recalculate(PhysicalParameters<T> params, T v_parall_begin, T v_parall_step, size_t v_parall_size) const {

		}

		InitialConditionsCalculation(ZFunc Z): Z(Z) { }

	private:
		ZFunc Z;
	};
} /*whif*/ } /*iki*/

#endif /*InitialParametersCalculationClass_CUH*/