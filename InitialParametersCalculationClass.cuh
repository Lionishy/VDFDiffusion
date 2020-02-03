#pragma once
#ifndef InitialParametersCalculationClass_CUH
#define InitialParametersCalculationClass_CUH

#include "SimpleTable.h"
#include "PhysicalParameters.h"
#include "ZFunc.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace iki { namespace whfi {
	template <typename T>
	struct ResonantVelocityEqn {
		__device__ T operator()(T omega) const {
			T Zc = Z(v_res - params.bulk_to_term_c)
				, Zh = Z(v_res * sqrt(params.TcTh_ratio) - params.bulk_to_term_h);
			return T(1. / 1836.)
				+ (omega - T(1)) * (omega - T(1)) / (v_res * params.betta_root_c) / (v_res * params.betta_root_c)
				- params.nc * ((omega * v_res) / (omega - T(1)) - params.bulk_to_term_c) * Zc
				- params.nh * ((omega * v_res * sqrt(params.TcTh_ratio)) / (omega - T(1)) - params.bulk_to_term_h) * Zh;
		}

		__device__ ResonantVelocityEqn(T v_res, iki::whfi::PhysicalParameters<T> params, iki::ZFunc<T> Z) : v_res(v_res), params(params), Z(Z) { }

		T v_res;
		PhysicalParameters<T> params;
		ZFunc<T> Z;
	};

	template <typename T>
	struct DispersionRootDerivative {
		__device__ T operator()(T omega, T k) const {
			T arg_c = (omega - T(1.)) / (k * params.betta_root_c) - params.bulk_to_term_c;
			T arg_h = (omega - T(1.)) / (k * params.betta_root_h) - params.bulk_to_term_h;
			T Zc = Z(arg_c), Zh = Z(arg_h);
			return params.nc / (k * params.betta_root_c) * (-Zc + (omega / (k * params.betta_root_c) - params.bulk_to_term_c) * (Zc * arg_c + T(1.)))
				+ params.nh / (k * params.betta_root_h) * (-Zh + (omega / (k * params.betta_root_h) - params.bulk_to_term_h) * (Zh * arg_h + T(1.)));
		}

		__device__ DispersionRootDerivative(iki::whfi::PhysicalParameters<T> params, iki::ZFunc<T> Z) : params(params), Z(Z) { }

		PhysicalParameters<T> params;
		ZFunc<T> Z;
	};


	template <typename T>
	__global__ void dispersion_relation_solve(T v_res_begin, T v_res_step, T *omega, T *derive, int *status, unsigned size, PhysicalParameters<T> params, T z_func_step, unsigned z_func_size, T *z_func_table) {
		unsigned shift = blockIdx.x * blockDim.x + threadIdx.x;
		T v_res = v_res_begin + shift * v_res_step;

		ResonantVelocityEqn<T> eqn(v_res, params, ZFunc<T>(z_func_step, z_func_size, z_func_table));
		DispersionRootDerivative<T> dispersion_derive(params, ZFunc<T>(z_func_step, z_func_size, z_func_table));

		*(status + shift) = step_solver(eqn, T(0.), T(1.e-5), T(1. + 1.e-6), omega + shift);
		if (0 == *(status + shift)) {
			T k = (*(omega + shift) - T(1.)) / (v_res * params.betta_root_c);
			*(derive + shift) = dispersion_derive(*(omega + shift), k);
		}
	}

	template <typename T>
	struct InitialParametersCalculationClass {

		void calc(size_t vperp_size, size_t vparall_size, UniformSpace<T,2u> v_space, PhysicalParameters<T> params) {
			dispersion_relation_solve <<<1, size>>> (v_res_dev, omegas_dev, derive_dev, status_dev, size, params, zfunc_table.space.axes[0].step, zfunc_table.bounds.components[0], zfunc_table_dev);
		}

		T *omega, *k, *derive; //device memory
	};
} /*whfi*/ } /*iki*/

#endif /*InitialParametersCalculationClass_CUH*/
