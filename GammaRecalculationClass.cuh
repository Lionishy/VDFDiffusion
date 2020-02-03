#include "DeviceMemory.h"
#include "SimpleTable.h"
#include "ZeroMoment.cuh"
#include "FirstMoment.cuh"
#include "AmplitudeKernel.cuh"
#include "DeviceException.h"

#include <vector>

namespace iki { namespace whfi {
	template <typename T>
	struct GammaReclaculation {

		GammaReclaculation(size_t vperp_size, size_t vparall_size, UniformSpace<T, 2u> velocity_space, T dt, std::vector<T> const &x_dfc_pivot_host, std::vector<T> const &y_dfc_pivot_host, std::vector<T> const &xy_dfc_pivot_host, std::vector<T> const &yx_dfc_pivot_host, std::vector<T> const &dispersion_derivative_host, std::vector<T> const &k_betta_host): 
			vperp_size(vperp_size), vparall_size(vparall_size), 
			velocity_space(velocity_space), 
			dt(dt), 
			vdf(nullptr), x_dfc(nullptr), y_dfc(nullptr), xy_dfc(nullptr), yx_dfc(nullptr), 
			internal_device_memory(4*vperp_size*vparall_size*sizeof(T) + 6*vparall_size*sizeof(T)) {

			//initial data copy
			{
				cudaError_t cudaStatus;
				if (cudaSuccess != (cudaStatus = cudaMemcpy(x_dfc_pivot, x_dfc_pivot_host.data(), vperp_size * vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(y_dfc_pivot, y_dfc_pivot_host.data(), vperp_size * vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(xy_dfc_pivot, xy_dfc_pivot_host.data(), vperp_size * vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(yx_dfc_pivot, yx_dfc_pivot_host.data(), vperp_size * vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);

				if (cudaSuccess != (cudaStatus = cudaMemcpy(dispersion_derivative, dispersion_derivative_host.data(), vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
				if (cudaSuccess != (cudaStatus = cudaMemcpy(k_betta, k_betta_host.data(), vparall_size * sizeof(T), cudaMemcpyHostToDevice))) throw DeviceException(cudaStatus);
			}
		}

		void external_memory_init(T const *vdf_ext, T *x_dfc_ext, T *y_dfc_ext, T *xy_dfc_ext, T *yx_dfc_ext) {
			vdf = vdf_ext;
			x_dfc = x_dfc_ext;
			y_dfc = y_dfc_ext;
			xy_dfc = xy_dfc_ext;
			yx_dfc = yx_dfc_ext;
		}

		size_t vperp_size, vparall_size; //vperp - x; vparall - y
		UniformSpace<T, 2u> velocity_space; //vperp - 0, vparall - 1
		T dt;

		T const *vdf;                       //external memory to be read
		T *x_dfc, *y_dfc, *xy_dfc, *yx_dfc; //external memory to be written

		DeviceMemory internal_device_memory;
		T *x_dfc_pivot, *y_dfc_pivot, *xy_dfc_pivot, *yx_dfc_pivot; //predefined data
		T *dispersion_derivative, *k_betta;                         //predefined data
		T *growth_rate_spectrum, *amplitude_spectrum, *zero_moment, *first_moment; //internal memory to be updated at every step
		
		

		void growth_rate_update() {
			math::device::zero_moment_kernel <<<1,vparall_size>>> (vdf, velocity_space.axes[0].begin, velocity_space.axes[0].step, vperp_size, vparall_size, zero_moment);
			math::device:first_moment_kernel <<<1,vparall_size>>> (vdf, velocity_space.axes[0].begin, velocity_space.axes[0].step, vperp_size, vparall_size, first_moment);

			device::gamma_kernel <<<1,vparall_size>>> (zero_moment, first_moment, k_betta, dispersion_derivative, velocity_space.axes[1].step, vparall_size, growth_rate_spectrum);

			device::amplitude_update_kernell <<<1,vparall_size>>> (growth_rate_spectrum, amplitude_spectrum, dt, vparall_size);
		}
	};

} /*whfi*/ } /*iki*/