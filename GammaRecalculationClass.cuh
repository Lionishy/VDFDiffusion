#include "DeviceMemory.h"
#include "SimpleTable.h"

#include "FirstMoment.cuh"





namespace iki { namespace whfi {
	template <typename T>
	struct GammaReclaculation {

		T const *vdf;                       //external memory to be read
		T *x_dfc, *y_dfc, *xy_dfc, *yx_dfc; //external memory to be written

		DeviceMemory internal_device_memory;
		T *dispersion_derivative, *k_betta;                         //readonly internal memory
		T *x_dfc_pivot, *y_dfc_pivot, *xy_dfc_pivot, *yx_dfc_pivot; //readonly internal memory
		T *growth_rate_spectrum, *amplitude_spectrum, *zero_moment, *first_moment; //internal memory to be updated at every step

		Bounds<2u> size;
		UniformSpace<T, 2u> velocity_space;

		void growth_rate_update() {
			zero_moment_kernel << <1, size.components[1] >> > (vdf, velocity_space.axes[0].begin, velocity_space.axes[0].step, size.components[0], size.components[1], zero_moment);
			first_moment_kernel <<<1, size.components[1]>>> (vdf, velocity_space.axes[0].begin, velocity_space.axes[0].step, size.components[0], size.components[1], first_moment);

			gamma_kernel << 1, size.components[1] >> > (zero_moment, first_index, k_betta, dispersion_derivative, velocity_space.axes[1].step, size.components[1], growth_rate_spectrum);
		}

		void amplitude_update() {

		}

	};

} /*whfi*/ } /*iki*/