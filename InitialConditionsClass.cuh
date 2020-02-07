#pragma once
#ifndef InitialConditionsClass_CUH
#define InitialConditionsClass_CUH

#include "ZFunc.h"
#include "SimpleTable.h"
#include "PhysicalParameters.h"
#include "StepSolver.h"
#include "ResonantVelocityEqn.h"
#include "DispersionRelationOmegaDerive.h"
#include "DispersionRelationKDerive.h"

#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

namespace iki { namespace whfi {
	template <typename T>
	struct InitialConditions {
		std::vector<T> omega, k_betta, dispersion_derivative, Dpr, Dpl, Dmx;
	};

	template <typename T>
	struct InitialConditionsCalculation {
		InitialConditions<T> recalculate(PhysicalParameters<T> params, T v_parall_begin, T v_parall_step, size_t v_parall_size) const {
			InitialConditions<T> result;
			DispersionRelationOmegaDerivative<T> dr_omega_derive(Z, params);
			DispersionRelationKDerivative<T> dr_k_derive(Z, params);

			for (size_t count = 0; count != v_parall_size; ++count) {
				T v_res = v_parall_begin + count * v_parall_step;
				try {
					T omega = math::step_solver(ResonantVelocityEqn<T>(v_res, params, Z), T(1.e-5), T(1.e-5), T(1. + 1.e-6));
					T k_betta = (omega - T(1.)) / v_res;
					T dispersion_derivative = dr_omega_derive(omega, k_betta);
					T group_velocity = -dr_k_derive(omega, k_betta) / dispersion_derivative;
					T Dpl = T(3.14159265) / params.betta_root_c / std::fabs(v_res - group_velocity / params.betta_root_c);
					T Dmx = Dpl / k_betta;
					T Dpr = Dmx / k_betta;

					result.omega.push_back(omega);
					result.k_betta.push_back(k_betta);
					result.dispersion_derivative.push_back(dispersion_derivative);
					result.Dpl.push_back(Dpl);
					result.Dmx.push_back(Dmx);
					result.Dpr.push_back(Dpr);
				}
				catch (std::exception const &ex) {
					std::cout << ex.what() << endl;
					std::stringstream error_stream;
					error_stream << "v_res = " << v_res << "  count = " << count;
					throw std::runtime_error(error_stream.str());
				}

				return result;
			}
		}

		InitialConditionsCalculation(ZFunc<T> Z): Z(Z) { }

	private:
		ZFunc<T> Z;
	};
} /*whif*/ } /*iki*/

#endif /*InitialParametersCalculationClass_CUH*/