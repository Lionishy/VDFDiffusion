#pragma once
#ifndef SimpleTwoDimensionalSolver_CUH
#define SimpleTwoDimensionalSolver_CUH

#include "AbstractTwoDimensionalSolver.cuh"

namespace iki { namespace diffusion { 
	template <typename T>
	struct SimpleTwoDimensionalSolver final : AbstractTwoDimensionalSolver<T> {

		SimpleTwoDimensionalSolver(size_t x_size, size_t y_size, T rx, T ry, std::vector<T> const &f, std::vector<T> const &x_dfc_host, std::vector<T> const &y_dfc_host, std::vector<T> const &xy_dfc_host, std::vector<T> const &yx_dfc_host): AbstractTwoDimensionalSolver<T>(x_size, y_size, rx, ry, f, x_dfc_host, y_dfc_host, xy_dfc_host, yx_dfc_host) { }

		void step() final {
			forward_step(x_dfc, y_dfc, xy_dfc, yx_dfc, rx, ry, x_size, y_size);
			cycle_transpose(x_size, y_size);

			correction_step(y_dfc, ry, x_size, y_size);
			cycle_transpose(y_size, x_size);

			forward_step_with_mixed_terms_correction(x_dfc, y_dfc, xy_dfc, yx_dfc, rx, ry, x_size, y_size);
			cycle_transpose(x_size, y_size);

			correction_step(y_dfc, ry, x_size, y_size);
			cycle_transpose(y_size, x_size);

			std::swap(f_prev_full, f_curr_full);
			std::swap(f_prev, f_curr);
		}
	};
} /* diffusion */ } /* iki */

#endif /* SimpleTwoDimensionalSolver_CUH */