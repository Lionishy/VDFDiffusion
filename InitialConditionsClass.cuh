#pragma once
#ifndef InitialConditionsClass_CUH
#define InitialConditionsClass_CUH

#include "ZFunc.h"
#include "SimpleTable.h"
#include "PhysicalParameters.h"
#include "StepSolver.h"
#include "ResonantVelocityEqn.h"

#include <vector>
#include <iterator>
#include <algorithm>

namespace iki { namespace whfi {
	template <typename T>
	struct InitialConditions {
		std::vector<T> omega, k_betta, dispersion_derivative, Dpr, Dpl, Dmx;
	};

	template <typename T>
	struct InitialConditionsCalculation {
		InitialConditions<T> recalculate(PhysicalParameters<T> params, T v_parall_begin, T v_parall_step, size_t v_parall_size) const {
			InitialConditions<T> result;
			for (size_t count = 0; count != v_parall_size; ++count) {
				result.omega.push_back(
					math::step_solver(ResonantVelocityEqn<T>(v_parall_begin + count * v_parall_step, params, Z), T(1.e-5), T(1.e-5), T(1. + 1.e-6))
				);
			}



		}

		InitialConditionsCalculation(ZFunc Z): Z(Z) { }

	private:
		ZFunc Z;
	};
} /*whif*/ } /*iki*/

#endif /*InitialParametersCalculationClass_CUH*/