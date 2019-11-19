#pragma once
#ifndef DiffusionRecalculationSolver_CUH
#define DiffusionRecalculationSolver_CUH

#include "AbstractTwoDimensionalSolver.cuh"
#include "DeviceMemory.h"

namespace iki { namespace diffusion {
	template <typename T>
	struct DiffusionRecalculationTwoDimensionalSolver: AbstractTwoDimensionalSolver<T> {
		DiffusionRecalculationTwoDimensionalSolver(size_t x_size, size_t y_size, T rx, T ry, std::vector<T> const &f_host, std::vector<T> const &x_dfc_host, std::vector<T> const &y_dfc_host, std::vector<T> const &xy_dfc_host, std::vector<T> const &yx_dfc_host, std::vector<T> const &gamma, std::vector<T> const &bk) : AbstractTwoDimensionalSolver(x_size, y_size, rx, ry, f_host, x_dfc_host, y_dfc_host, xy_dfc_host, yx_dfc_host), device_ptr(2 * y_size * sizeof(T) + 4 * x_size * y_size * sizeof(T))	{
			Bk = (T*)device_ptr.get();
			gamma = Bk + y_size;
			x_dfc_pvt = gamma + y_size;
			y_dfc_pvt = x_dfc_pvt + x_size * y_size;
			xy_dfc_pvt = y_dfc_pvt + x_size * y_size;
			yx_dfc_pvt = xy_dfc_pvt + x_size * y_size;
		}

		void diffusion_coefficients_recalculation() {

		}

		DeviceMemory device_ptr;
		float *Bk, *gamma;
		float *x_dfc_pvt, *y_dfc_pvt, *xy_dfc_pvt, *yx_dfc_pvt;

	};
} /* diffusion */ } /* iki */

#endif /* DiffusionRecalculationSolver_CUH */